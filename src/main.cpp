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
        std::cerr << "Usage: ./autoencoder <cifar-10-batches-bin> [--debug {true|false}] [--keep-10-percent {true|false}]\n";
        return 1;
    }

    std::string cifar_dir = argv[1];

    //ARGUMENT-PARSING SECTION
    /*
    --debug:
        - true: only run 2 epochs, each using 2 first batches (32 samples each)
        - false: run up to 20 epochs, with full datasets
    --keep-10-percent
        - true: only retain 5000 training samples and 1000 testing samples
        - false: retain entire dataset (50000 training samples, 10000 testing samples)
    */
    bool debug = false;
    bool keep_10_percent = true;
    
    int i = 2;
    while (i < argc) {
        std::string arg = argv[i];
        
        if (arg == "--debug") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --debug requires a value (true|false)\n";
                return 1;
            }
            std::string debug_str = argv[i + 1];
            if (debug_str == "true") {
                debug = true;
            } else if (debug_str == "false") {
                debug = false;
            } else {
                std::cerr << "Error: --debug parameter must be 'true' or 'false', got '" << debug_str << "'\n";
                return 1;
            }
            i += 2;
        } else if (arg == "--keep-10-percent") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --keep-10-percent requires a value (true|false)\n";
                return 1;
            }
            std::string keep_str = argv[i + 1];
            if (keep_str == "true") {
                keep_10_percent = true;
            } else if (keep_str == "false") {
                keep_10_percent = false;
            } else {
                std::cerr << "Error: --keep-10-percent parameter must be 'true' or 'false', got '" << keep_str << "'\n";
                return 1;
            }
            i += 2;
        } else {
            std::cerr << "Error: Unknown argument '" << arg << "'\n";
            return 1;
        }
    }

    std::cout << "Loading CIFAR-10...\n";
    CIFAR10 ds;
    ds.load(cifar_dir, keep_10_percent);

    int Ntrain = ds.train_size();
    int Ntest  = ds.test_size();
    std::cout << "Total images: " << (Ntrain + Ntest)
              << " (train " << Ntrain << ", test " << Ntest << ")\n";

    // Hyperparams
    int batch_size  = 32;
    int epochs = debug ? 2 : 20;
    float lr   = 1e-3;

    Autoencoder ae;
    MSELoss criterion;

    std::filesystem::create_directory("out");

    // Create data loaders
    DataLoader train_loader(ds.train_images(), ds.train_labels(), batch_size, true);   // shuffle
    DataLoader test_loader(ds.test_images(), ds.test_labels(), batch_size, false);     // no shuffle

    // DEBUG: limit batches
    int train_max_batches = debug ? 2 : INT_MAX;

    if (debug) {
        std::cout << "\n=== DEBUG MODE: Running with 2 batches per epoch ===\n\n";
    }
    else {
        std::cout << "\n=== NON-DEBUG MODE ===\n\n";
    }

    // =========================
    // Training Loop
    // =========================
    for (int ep = 1; ep <= epochs; ++ep) {
        train_loader.reset(true);  // reset and reshuffle
        
        auto t0 = std::chrono::high_resolution_clock::now();
        double epoch_loss = 0.0;
        int train_nb = 0;

        while (train_loader.has_next() && train_nb < train_max_batches) { // DEBUG: train_nb < train_max_batches to limit only 2 batches per epoch
            auto batch_data = train_loader.next();
            const Tensor& x = batch_data.images;

            // print progress occasionally
            if (train_nb > 0 && train_nb % 16 == 0) {
                std::cout << "[Epoch " << ep
                          << "] batch " << train_nb << "/" << train_loader.num_batches() << "\n";
            }

            // ----- FORWARD -----
            const Tensor& y = ae.forward(x);

            // ----- LOSS + dLoss/dY -----
            auto [loss, dY] = criterion.forward_backward(y, x);
            epoch_loss += loss;
            ++train_nb;

            // ----- BACKWARD + UPDATE -----
            ae.backward_and_update(dY, lr);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "Epoch " << ep
                  << " | train_loss=" << (epoch_loss / train_nb)
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
    // Final Test Evaluation
    // =========================
    std::cout << "\n=== Final Test Evaluation ===\n";
    test_loader.reset(false);
    double test_loss = 0.0;
    int test_nb = 0;
    int test_max_batches = debug ? 2 : INT_MAX;

    while (test_loader.has_next() && test_nb < test_max_batches) {
        auto batch_data = test_loader.next();
        const Tensor& x_test = batch_data.images;

        const Tensor& y_test = ae.forward(x_test);
        auto [loss, dY] = criterion.forward_backward(y_test, x_test);
        test_loss += loss;
        ++test_nb;
    }

    std::cout << "Final test_loss=" << (test_loss / test_nb) << "\n";

    std::cout << "Done. Check ./out/*.ppm\n";
    return 0;
}
