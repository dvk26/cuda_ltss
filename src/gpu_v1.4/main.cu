#include "../include/dataset.hpp"
#include "gpu_autoencoder.hpp"
#include <iostream>
#include <chrono>
#include <filesystem>
#include <iomanip> // Để format in số cho đẹp
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    // 1. Xử lý tham số đầu vào
    if (argc < 2) {
        std::cerr << "Usage: ./autoencoder_gpu_v1.4 <cifar-10-batches-bin> [--keep-partial]\n";
        return 1;
    }

    std::string cifar_dir = argv[1];
    bool keep_10_percent = false;

    int i = 2;
    while (i < argc) {
        std::string arg = argv[i];
        if (arg == "--keep-partial") {
            keep_10_percent = true;
            i += 1;
        } else {
            std::cerr << "Error: Unknown argument '" << arg << "'\n";
            return 1;
        }
    }

    // 2. Load Dữ liệu
    std::cout << "Loading CIFAR-10...\n";
    CIFAR10 ds;
    ds.load(cifar_dir, keep_10_percent);

    int Ntrain = ds.train_size();
    int Ntest  = ds.test_size();
    std::cout << "Total images: " << (Ntrain + Ntest)
              << " (train " << Ntrain << ", test " << Ntest << ")\n";

    // 3. Cấu hình Hyperparams
    // Với v1.4 (Fusion + Shared Mem), bộ nhớ được tiết kiệm nên có thể tăng batch_size
    // Tuy nhiên giữ 64 để so sánh tốc độ với bản cũ cho khách quan.
    int batch_size = 128;
    int epochs = 20;
    float lr = 1e-3f;

    // Tạo thư mục output
    const std::string out_dir = "out-gpu-v1.4";
    fs::create_directories(out_dir);

    // 4. Khởi tạo Model
    // Lưu ý: Không cần MSELoss criterion của CPU nữa!
    GPUAutoencoder gpu_ae(batch_size, 32, 32);

    // Create data loaders
    DataLoader train_loader(ds.train_images(), ds.train_labels(), batch_size, true);   // shuffle
    DataLoader test_loader(ds.test_images(), ds.test_labels(), batch_size, false);     // no shuffle

    std::cout << "Starting training v1.4...\n";

    cudaStream_t stream_h2d = nullptr;
    cudaStream_t stream_compute = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream_h2d));
    CUDA_CHECK(cudaStreamCreate(&stream_compute));
    gpu_ae.set_compute_stream(stream_compute);

    cudaEvent_t h2d_ready[2];
    CUDA_CHECK(cudaEventCreateWithFlags(&h2d_ready[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&h2d_ready[1], cudaEventDisableTiming));

    size_t batch_elems = static_cast<size_t>(batch_size) * 3 * 32 * 32;
    size_t batch_bytes = batch_elems * sizeof(float);
    float* h_pinned[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaMallocHost(&h_pinned[0], batch_bytes));
    CUDA_CHECK(cudaMallocHost(&h_pinned[1], batch_bytes));

    // =========================
    // Training Loop
    // =========================
    for (int ep = 1; ep <= epochs; ++ep) {
        train_loader.reset(true);

        auto t0 = std::chrono::high_resolution_clock::now();
        double epoch_loss = 0.0;
        int train_nb = 0;
        int num_batches = train_loader.num_batches();
        size_t peak_used_bytes = 0;

        auto stage_next_batch = [&](int buffer_index) -> bool {
            while (train_loader.has_next()) {
                auto batch_data = train_loader.next();
                const Tensor& x = batch_data.images;
                if (x.N() != batch_size) {
                    continue;
                }
                std::memcpy(h_pinned[buffer_index], x.raw().data(), batch_bytes);
                gpu_ae.stage_input_async(h_pinned[buffer_index], buffer_index, stream_h2d);
                CUDA_CHECK(cudaEventRecord(h2d_ready[buffer_index], stream_h2d));
                return true;
            }
            return false;
        };

        bool have_curr = stage_next_batch(0);
        bool have_next = stage_next_batch(1);
        int cur = 0;
        int next = 1;

        while (have_curr) {
            CUDA_CHECK(cudaStreamWaitEvent(stream_compute, h2d_ready[cur], 0));
            gpu_ae.set_active_input_buffer(cur);

            float loss = gpu_ae.train_step_device(lr);

            epoch_loss += loss;
            ++train_nb;

            size_t free_bytes = 0;
            size_t total_bytes = 0;
            cudaMemGetInfo(&free_bytes, &total_bytes);
            peak_used_bytes = std::max(peak_used_bytes, total_bytes - free_bytes);

            if (train_nb % 10 == 0 || train_nb == num_batches) {
                std::cout << "\r[Epoch " << ep << "/" << epochs << "] "
                          << "Batch " << train_nb << "/" << num_batches 
                          << " | Loss: " << std::fixed << std::setprecision(4) << (epoch_loss / train_nb) 
                          << std::flush;
            }

            if (!have_next) {
                break;
            }

            int old = cur;
            cur = next;
            next = old;
            have_curr = true;
            have_next = stage_next_batch(next);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();

        // Xuống dòng sau khi chạy xong epoch
        std::cout << "\n   -> Time: " << sec << "s (" 
                  << std::fixed << std::setprecision(1) << (train_nb / sec) << " batches/s)\n";
        std::cout << "   -> Peak VRAM: " << std::fixed << std::setprecision(2)
                  << (peak_used_bytes / (1024.0 * 1024.0)) << " MiB\n";

        // Save weights
        CUDA_CHECK(cudaStreamSynchronize(stream_compute));
        gpu_ae.save_weights(out_dir + "/weights_epoch_" + std::to_string(ep) + ".bin");
    }

    // =========================
    // Final Test Evaluation
    // =========================
    std::cout << "\n=== Final Test Evaluation ===\n";
    test_loader.reset(false);
    double test_loss = 0.0;
    int test_nb = 0;

    while (test_loader.has_next()) {
        auto batch_data = test_loader.next();
        const Tensor& x_test = batch_data.images;

        if (x_test.N() != batch_size) continue;

        // Dùng compute_loss (chạy hoàn toàn trên GPU)
        float loss = gpu_ae.compute_loss(x_test);
        
        test_loss += loss;
        ++test_nb;
    }

    std::cout << "Final Test MSE Loss: " << (test_loss / test_nb) << "\n";
    std::cout << "Done. Check ./" << out_dir << " for weights.\n";

    CUDA_CHECK(cudaFreeHost(h_pinned[0]));
    CUDA_CHECK(cudaFreeHost(h_pinned[1]));
    CUDA_CHECK(cudaEventDestroy(h2d_ready[0]));
    CUDA_CHECK(cudaEventDestroy(h2d_ready[1]));
    CUDA_CHECK(cudaStreamDestroy(stream_h2d));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));

    cudaDeviceReset();
    return 0;
}
