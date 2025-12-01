#include <iostream>
#include <chrono>
#include <random>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <climits>

#include "dataset.hpp"
#include "autoencoder.hpp"
#include "layers.hpp"

// ============================================
// Save image as PPM
// ============================================
static void save_ppm(const std::string& path, const Tensor& imgNCHW, int n=0) {
    int H = imgNCHW.H(), W = imgNCHW.W();
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << W << " " << H << "\n255\n";

    for (int h=0; h<H; ++h)
    for (int w=0; w<W; ++w) {
        auto clamp01 = [](float v){ return std::max(0.f, std::min(1.f, v)); };
        unsigned char r = (unsigned char)std::round(clamp01(imgNCHW.at(n,0,h,w))*255.f);
        unsigned char g = (unsigned char)std::round(clamp01(imgNCHW.at(n,1,h,w))*255.f);
        unsigned char b = (unsigned char)std::round(clamp01(imgNCHW.at(n,2,h,w))*255.f);
        f.put(r); f.put(g); f.put(b);
    }
}

// ============================================
// MAIN
// ============================================
int main(int argc, char** argv){
    if (argc < 2) {
        std::cerr << "Usage: ./autoencoder <cifar-10-batches-bin>\n";
        return 1;
    }

    std::string cifar_dir = argv[1];

    std::cout << "Loading CIFAR-10...\n";
    CIFAR10 ds;
    ds.load(cifar_dir);

    int Ntrain = ds.train_size();
    int Ntest  = ds.test_size();
    std::cout << "Total images: " << (Ntrain + Ntest)
              << " (train " << Ntrain << ", test " << Ntest << ")\n";

    // Hyperparams
    int batch  = 32;
    int epochs = 5;
    float lr   = 1e-3;

    Autoencoder ae;
    MSELoss criterion;

    std::filesystem::create_directory("out");

    // Create data loaders
    DataLoader train_loader(ds.train_images(), ds.train_labels(), batch, true);   // shuffle
    DataLoader test_loader(ds.test_images(), ds.test_labels(), batch, false);     // no shuffle

    // DEBUG: limit to 1 batch for testing
    bool debug = false;  // Set to true to debug with only 1 batch
    int max_batches = debug ? 1 : INT_MAX;

    if (debug) {
        std::cout << "\n=== DEBUG MODE: Running with 1 batch only ===\n\n";
    }

    // =========================
    // Training Loop
    // =========================
    for (int ep = 1; ep <= epochs; ++ep) {
        train_loader.reset(true);  // reset and reshuffle
        
        auto t0 = std::chrono::high_resolution_clock::now();
        double epoch_loss = 0.0;
        int nb = 0;

        while (train_loader.has_next() && nb < max_batches) { // DEBUG: nb < max_batches to limit only 1 batch per epoch
            auto batch_data = train_loader.next();
            const Tensor& x = batch_data.images;

            // print progress occasionally
            if (nb > 0 && nb % 16 == 0) {
                std::cout << "[Epoch " << ep
                          << "] batch " << nb << "/" << train_loader.num_batches() << "\n";
            }

            // ----- FORWARD -----
            const Tensor& y = ae.forward(x);

            // ----- LOSS + dLoss/dY -----
            auto [loss, dY] = criterion.forward_backward(y, x);
            epoch_loss += loss;
            ++nb;

            // ----- BACKWARD + UPDATE -----
            ae.backward_and_update(dY, lr);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "Epoch " << ep
                  << " | train_loss=" << (epoch_loss / nb)
                  << " | time=" << sec << "s\n";

        // Lưu weights sau mỗi epoch
        ae.save_weights("out/weights_epoch_" + std::to_string(ep) + ".bin");

        // --------------------------
        // dump reconstruction example
        // --------------------------
        Tensor one(1,3,32,32);
        for (int c=0; c<3; ++c)
        for (int h=0; h<32; ++h)
        for (int w=0; w<32; ++w)
            one.at(0,c,h,w) = ds.train_images().at(0,c,h,w);

        const Tensor& recon = ae.forward(one);

        save_ppm("out/ep" + std::to_string(ep) + "_orig.ppm", one);
        save_ppm("out/ep" + std::to_string(ep) + "_recon.ppm", recon);
    }

    // =========================
    // Final Test Evaluation (Bring out the code inside true branch if no need for DEBUG)
    // =========================
    if (!debug) {
        std::cout << "\n=== Final Test Evaluation ===\n";
        test_loader.reset(false);
        double test_loss = 0.0;
        int test_nb = 0;

        while (test_loader.has_next()) {
            auto batch_data = test_loader.next();
            const Tensor& x_test = batch_data.images;

            const Tensor& y_test = ae.forward(x_test);
            auto [loss, dY] = criterion.forward_backward(y_test, x_test);
            test_loss += loss;
            ++test_nb;
        }

        std::cout << "Final test_loss=" << (test_loss / test_nb) << "\n";
    } else {
        std::cout << "\n(Skipping test evaluation in debug mode)\n";
    }

    std::cout << "Done. Check ./out/*.ppm\n";
    return 0;
}
