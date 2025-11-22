#include <iostream>
#include <chrono>
#include <random>
#include <numeric>
#include <filesystem>
#include <fstream>

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

    const Tensor& all = ds.images();
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

    // shuffled index list
    std::vector<int> idx(Ntrain);
    std::iota(idx.begin(), idx.end(), 0);

    // =========================
    // Training Loop
    // =========================
    for (int ep = 1; ep <= epochs; ++ep) {

        std::mt19937 rng(1234 + ep);
        std::shuffle(idx.begin(), idx.end(), rng);

        auto t0 = std::chrono::high_resolution_clock::now();
        double epoch_loss = 0.0;
        int nb = 0;

        for (int i = 0; i < Ntrain; i += batch) {
            int cur = std::min(batch, Ntrain - i);

            // mini-batch tensor
            Tensor x(cur, 3, 32, 32);

            // print progress occasionally
            if (i > 0 && i % 512 == 0) {
                std::cout << "[Epoch " << ep
                          << "] processing batch at index " << i << "\n";
            }

            // copy data from dataset → x
            for (int k = 0; k < cur; ++k) {
                int id = idx[i + k];
                for (int c = 0; c < 3; ++c)
                for (int h = 0; h < 32; ++h)
                for (int w = 0; w < 32; ++w)
                    x.at(k, c, h, w) = all.at(id, c, h, w);
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
                  << " | loss=" << (epoch_loss / nb)
                  << " | time=" << sec << "s\n";

        // --------------------------
        // dump reconstruction example
        // --------------------------
        Tensor one(1,3,32,32);
        for (int c=0; c<3; ++c)
        for (int h=0; h<32; ++h)
        for (int w=0; w<32; ++w)
            one.at(0,c,h,w) = all.at(0,c,h,w);

        const Tensor& recon = ae.forward(one);

        save_ppm("out/ep" + std::to_string(ep) + "_orig.ppm", one);
        save_ppm("out/ep" + std::to_string(ep) + "_recon.ppm", recon);
    }

    std::cout << "Done. Check ./out/*.ppm\n";
    return 0;
}
