#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <random>
#include <fstream>
#include "../include/tensor.hpp"
#include "../include/autoencoder.hpp" 

// =====================
// CUDA CHECK MACRO
// =====================
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            throw std::runtime_error(                                         \
                std::string("CUDA error: ") + cudaGetErrorString(err) +       \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
        }                                                                     \
    } while (0)

inline size_t nchw_size(int N, int C, int H, int W) {
    return static_cast<size_t>(N) * C * H * W;
}

// =====================
// GPU Autoencoder (Optimized)
// Enc: 3->256 -> pool2 -> 256->128 -> pool2
// Dec: 128->128 -> up2 -> 128->256 -> up2 -> 256->3
// =====================

class GPUAutoencoder {
public:
    GPUAutoencoder(int batch_size, int H = 32, int W = 32);
    ~GPUAutoencoder();

    // Encode only: x [N,3,H,W] -> latent [N,128,H/4,W/4]
    Tensor encode(const Tensor& x_host);

    // Decode only: z [N,128,H/4,W/4] -> y [N,3,H,W]
    Tensor decode(const Tensor& z_host);

    // Forward: x -> encode -> decode -> y (inference only, no backprop)
    Tensor forward(const Tensor& x_host);

    // Backward + SGD update: backprop from d_dy_ and update weights
    void backward_and_update(float lr);

    // Compute MSE Loss and its gradient
    float compute_loss();

    // Save weights directly from GPU to file
    void save_weights(const std::string& path) const;

private:
    int N_;        // batch size
    int H_, W_;
    int H1_, W1_;
    int H2_, W2_;

    // ---------- Weights ----------
    // Conv1: 3 -> 256
    float* d_w1_; float* d_b1_; float* d_gw1_; float* d_gb1_;
    // Conv2: 256 -> 128
    float* d_w2_; float* d_b2_; float* d_gw2_; float* d_gb2_;
    // Conv3: 128 -> 128
    float* d_w3_; float* d_b3_; float* d_gw3_; float* d_gb3_;
    // Conv4: 128 -> 256
    float* d_w4_; float* d_b4_; float* d_gw4_; float* d_gb4_;
    // Conv5: 256 -> 3
    float* d_w5_; float* d_b5_; float* d_gw5_; float* d_gb5_;

    // ---------- Activations ----------
    float* d_x_;      // [N,3,32,32]
    float* d_c1_;     // [N,256,32,32]
    float* d_r1_;     // [N,256,32,32]
    float* d_p1_;     // [N,256,16,16]

    float* d_c2_;     // [N,128,16,16]
    float* d_r2_;     // [N,128,16,16]
    float* d_p2_;     // [N,128,8,8]  // latent

    float* d_c3_;     // [N,128,8,8]
    float* d_r3_;     // [N,128,8,8]
    float* d_u1_;     // [N,128,16,16]

    float* d_c4_;     // [N,256,16,16]
    float* d_r4_;     // [N,256,16,16]
    float* d_u2_;     // [N,256,32,32]

    float* d_c5_;     // [N,3,32,32]  // output

    // ---------- Pooling mask ----------
    int* d_mask1_;    // [N,256,16,16]
    int* d_mask2_;    // [N,128,8,8]

    // ---------- Gradients wrt activations ----------
    float* d_dc5_;    // dL/dc5
    float* d_du2_;
    float* d_dr4_;
    float* d_dc4_;
    float* d_du1_;
    float* d_dr3_;
    float* d_dc3_;
    float* d_dp2_;
    float* d_dr2_;
    float* d_dc2_;
    float* d_dp1_;
    float* d_dr1_;
    float* d_dc1_;
    float* d_dx_;     // dL/dx

    float* d_dy_;         // Chứa gradient (Pred - Target) tính ngay trên GPU
    float* d_loss_accum_; // Buffer 1 float để tích lũy Loss

    // ---------- Helpers ----------
    void alloc_all();
    void free_all();
    void init_weights_random();

    void encode_no_copy();
    void decode_no_copy();

    // Không cho copy
    GPUAutoencoder(const GPUAutoencoder&) = delete;
    GPUAutoencoder& operator=(const GPUAutoencoder&) = delete;
};
