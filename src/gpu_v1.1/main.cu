#include "../include/dataset.hpp"
#include "gpu_autoencoder.hpp"
#include <iostream>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./gpu-autoencoder <cifar-10-batches-bin> [--keep-partial]\n";
        return 1;
    }

    std::string cifar_dir = argv[1];
    bool keep_four_percent = false;

    int i = 2;
    while (i < argc) {
        std::string arg = argv[i];
        if (arg == "--keep-partial") {
            keep_four_percent = true;
            i += 1;
        } else {
            std::cerr << "Error: Unknown argument '" << arg << "'\n";
            return 1;
        }
    }

    std::cout << "Loading CIFAR-10...\n";
    CIFAR10 ds;
    ds.load(cifar_dir, keep_four_percent);

    int Ntrain = ds.train_size();
    int Ntest  = ds.test_size();
    std::cout << "Total images: " << (Ntrain + Ntest)
              << " (train " << Ntrain << ", test " << Ntest << ")\n";

    // Hyperparams
    int batch_size = 64;
    int epochs = 20;
    float lr = 1e-3f;

    // Output directory
    const std::string out_dir = "out-gpu";
    fs::create_directories(out_dir);

    GPUAutoencoder gpu_ae(batch_size, 32, 32);

    // Create data loaders
    DataLoader train_loader(ds.train_images(), ds.train_labels(), batch_size, true);   // shuffle
    DataLoader test_loader(ds.test_images(), ds.test_labels(), batch_size, false);     // no shuffle

    // =========================
    // Training Loop
    // =========================
    for (int ep = 1; ep <= epochs; ++ep) {
        train_loader.reset(true);

        auto t0 = std::chrono::high_resolution_clock::now();
        double epoch_loss = 0.0;
        int train_nb = 0;
        int num_batches = train_loader.num_batches();

        while (train_loader.has_next()) {
            auto batch_data = train_loader.next();
            const Tensor& x = batch_data.images;

            // Skip incomplete batches
            if (x.N() != batch_size) {
                continue;
            }

            // ----- FORWARD -----
            Tensor y = gpu_ae.forward(x);

            // ----- COMPUTE LOSS & GRADIENT (gradient is saved inside Autoencoder class) -----
            float loss = gpu_ae.compute_loss();
            epoch_loss += loss;
            ++train_nb;
            // ----- BACKWARD + UPDATE -----
            gpu_ae.backward_and_update(lr);

            // Print progress (cập nhật trên cùng 1 dòng cho gọn)
            if (train_nb % 10 == 0 || train_nb == num_batches) {
                std::cout << "\r[Epoch " << ep << "/" << epochs << "] "
                          << "Batch " << train_nb << "/" << num_batches 
                          << " | Loss: " << std::fixed << std::setprecision(4) << (epoch_loss / train_nb) 
                          << std::flush;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "\n=== Epoch " << ep
                  << " | train_loss=" << (epoch_loss / train_nb)
                  << " | time=" << sec << "s ===\n";

        // Save weights directly from GPU
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

        // Skip incomplete batches
        if (x_test.N() != batch_size) {
            continue;
        }

        Tensor y_test = gpu_ae.forward(x_test);
        float loss = gpu_ae.compute_loss();
        test_loss += loss;
        ++test_nb;
    }

    std::cout << "Final test_loss=" << (test_loss / test_nb) << "\n";
    std::cout << "Done. Check ./" << out_dir << "/*.bin for weights.\n";

    cudaDeviceReset();
    return 0;
}
