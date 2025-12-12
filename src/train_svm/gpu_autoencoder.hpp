#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <random>
#include <fstream>
#include "../include/tensor.hpp"

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
// GPU Autoencoder (Optimized v1.4)
// Pipeline: Full GPU (Forward -> Loss -> Backward)
// =====================

class GPUAutoencoder {
public:
    GPUAutoencoder(int batch_size, int H = 32, int W = 32);
    ~GPUAutoencoder();

    // --- API v1.4: Full GPU Training Pipeline ---
    // Chạy toàn bộ 1 bước train trên GPU: Forward -> MSE Loss -> Backward -> Update
    // Trả về: Giá trị Loss (float) để hiển thị
    float train_step(const Tensor& x_host, float lr);

    // Tính loss trên tập test (chỉ Forward + MSE)
    float compute_loss(const Tensor& x_host);

    // --- API Legacy / Inference ---
    Tensor encode(const Tensor& x_host);
    Tensor decode(const Tensor& z_host);
    Tensor forward(const Tensor& x_host);
    
    // Hàm này hỗ trợ cả legacy (CPU grad) và v1.4 (GPU grad)
    void backward_and_update(const Tensor& dOut, float lr);

    void save_weights(const std::string& path) const;
    void load_weights(const std::string& path);

private:
    int N_;        // batch size
    int H_, W_;
    int H1_, W1_;
    int H2_, W2_;

    // ---------- Weights ----------
    float *d_w1_, *d_b1_, *d_gw1_, *d_gb1_;
    float *d_w2_, *d_b2_, *d_gw2_, *d_gb2_;
    float *d_w3_, *d_b3_, *d_gw3_, *d_gb3_;
    float *d_w4_, *d_b4_, *d_gw4_, *d_gb4_;
    float *d_w5_, *d_b5_, *d_gw5_, *d_gb5_;

    // ---------- Activations ----------
    float* d_x_;      // Input [N,3,32,32]
    
    // Encoder
    float* d_c1_; float* d_r1_; float* d_p1_;
    float* d_c2_; float* d_r2_; float* d_p2_; // Latent

    // Decoder
    float* d_c3_; float* d_r3_; float* d_u1_;
    float* d_c4_; float* d_r4_; float* d_u2_;
    float* d_c5_;     // Output [N,3,32,32]

    // ---------- Pooling mask ----------
    int* d_mask1_;
    int* d_mask2_;

    // ---------- Gradients ----------
    float* d_dc5_;    // dL/dc5 (Gradient tại output layer)
    float* d_du2_; float* d_dr4_; float* d_dc4_;
    float* d_du1_; float* d_dr3_; float* d_dc3_;
    float* d_dp2_; float* d_dr2_; float* d_dc2_;
    float* d_dp1_; float* d_dr1_; float* d_dc1_;
    float* d_dx_;     // dL/dx

    // ---------- [NEW] Internal Buffers for v1.4 ----------
    float* d_dy_;         // Chứa gradient (Pred - Target) tính ngay trên GPU
    float* d_loss_accum_; // Buffer 1 float để tích lũy Loss

    // ---------- Helpers ----------
    void alloc_all();
    void free_all();
    void init_weights_random();

    void set_input(const Tensor& x_host);
    void forward_pass();
    
    // Tham số d_dy_host_ptr có thể là nullptr nếu dùng pipeline v1.4
    void backward_pass(const float* d_dy_host_ptr, float lr);

    // Disable copy
    GPUAutoencoder(const GPUAutoencoder&) = delete;
    GPUAutoencoder& operator=(const GPUAutoencoder&) = delete;
};