#include "dataset.hpp"
#include "gpu_autoencoder.hpp"
#include <iostream>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <iomanip>


namespace fs = std::filesystem;

int main() {
    // Nếu train trên Colab + Drive:
    // nhớ mount: drive.mount('/content/drive')
    const std::string ckpt_dir = "/content/drive/MyDrive/ltss_autoencoder_gpu_out";

    // Tạo folder lưu weights (nếu chưa có)
    fs::create_directories(ckpt_dir);

    CIFAR10 ds("cifar-10-batches-bin");
    bool debug = true;
    int batch_size = 64;
    int epochs = 5;
    float lr = 1e-3f;

    DataLoader train_loader(ds.train_images(), ds.train_labels(), batch_size, true);
    DataLoader test_loader(ds.test_images(), ds.test_labels(), batch_size, false);

    GPUAutoencoder gpu_ae(batch_size, 32, 32);

    // Model CPU dùng làm "vỏ" để lưu/đọc weights
    Autoencoder cpu_ae;

    for (int ep = 0; ep < epochs; ++ep) {
        train_loader.reset();
        int it = 0;

        double epoch_loss = 0.0;
        int nb = 0;

        auto t0 = std::chrono::high_resolution_clock::now();

        while (train_loader.has_next()) {
            auto batch_data = train_loader.next();
            const Tensor& x = batch_data.images;

            if (x.N() != batch_size) break; // bỏ batch cuối nếu lẻ

            float loss = gpu_ae.train_batch(x, lr);
            epoch_loss += loss;
            ++nb;
            ++it;

            if (it % 50 == 0) {
                std::cout << "[epoch " << ep << "] iter " << it
                          << ", loss=" << loss << "\n";
            }

            if (debug && it >= 2) break;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1 - t0).count();

        double avg_loss = epoch_loss / std::max(1, nb);
        std::cout << "Epoch " << ep << " done. Avg loss="
                  << avg_loss
                  << ", time=" << ms << " ms\n";

        // ====== LƯU WEIGHTS SAU MỖI EPOCH ======
        // 1) Copy weights từ GPU -> CPU
        gpu_ae.copy_weights_to_cpu(cpu_ae);

        // 2) Tạo tên file: .../weights_epoch_000.bin, 001.bin, ...
        std::ostringstream oss;
        oss << ckpt_dir << "/weights_epoch_"
            << std::setw(3) << std::setfill('0') << ep << ".bin";
        std::string weight_path = oss.str();

        // 3) Lưu weights
        cpu_ae.save_weights(weight_path);

        std::cout << "  Saved weights to: " << weight_path << "\n";
    }

    // Test: reconstruct vài ảnh
    DataLoader test_loader2(ds.test_images(), ds.test_labels(), batch_size, false);
    auto batch = test_loader2.next();
    Tensor recon;
    gpu_ae.forward(batch.images, recon);

    // TODO: save_ppm(...) giống CPU version

    cudaDeviceReset();
    return 0;
}
