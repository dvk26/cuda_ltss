#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <random>
#include <fstream>
#include <cstddef>
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
    float* d_x_;       // Input (FP32) [N,3,H,W]
    __half* d_xh_;     // Input (FP16) [N,3,H,W]

    // Encoder (FP16 activations; ReLU output stored directly in d_c*)
    __half* d_c1_; __half* d_p1_;
    __half* d_c2_; __half* d_p2_; // Latent

    // Decoder (FP16 activations; final output kept FP32 for loss/metrics)
    __half* d_c3_; __half* d_u1_;
    __half* d_c4_; __half* d_u2_;
    float* d_c5_;      // Output (FP32) [N,3,H,W]

    // Scratch buffers for checkpointing / recomputation.
    // Layout (shared by alias pointers above):
    // - d_scratch256_hw_:  [N,256,H,W]  reused for `c1` and `u2`
    // - d_scratch256_h1w1_: [N,256,H1,W1] reused for `c4`
    // - d_scratch128_h1w1_: [N,128,H1,W1] reused for `c2` and `u1`
    // - d_scratch128_h2w2_: [N,128,H2,W2] reused for `c3`
    __half* d_scratch256_hw_;
    __half* d_scratch256_h1w1_;
    __half* d_scratch128_h1w1_;
    __half* d_scratch128_h2w2_;

    // ---------- Pooling mask ----------
    int* d_mask1_;
    int* d_mask2_;

    // ---------- Gradients ----------
    // [v1.4] Ping-pong gradient pools:
    // - d_grad_pool_a_/b_ are the owners (the only pointers that are cudaFree'd).
    // - The named gradients (d_dc5_, d_du2_, ...) are aliases into one of the pools.
    float* d_grad_pool_a_ = nullptr;
    float* d_grad_pool_b_ = nullptr;
    size_t grad_pool_elems_ = 0;

    float* d_dc5_;    // dL/dc5 (Gradient tại output layer)
    float* d_du2_; float* d_dr4_; float* d_dc4_;
    float* d_du1_; float* d_dr3_; float* d_dc3_;
    float* d_dp2_; float* d_dr2_; float* d_dc2_;
    float* d_dp1_; float* d_dr1_; float* d_dc1_;
    float* d_dx_;     // dL/dx

    // ---------- [NEW] Internal Buffers for v1.4 ----------
    float* d_dy_;         // Chứa gradient (Pred - Target) tính ngay trên GPU
    float* d_loss_accum_; // Buffer 1 float để tích lũy Loss

    // ---------- Dynamic loss scaling / safety ----------
    int* d_found_inf_nan_;
    float loss_scale_;
    float loss_scale_min_;
    float loss_scale_max_;
    float loss_scale_growth_factor_;
    float loss_scale_backoff_factor_;
    int loss_scale_growth_interval_steps_;
    int loss_scale_good_steps_;

    // ---------- Manual checkpointing ----------
    enum class CheckpointMode { none, stage_boundaries };
    CheckpointMode checkpoint_mode_;

    // ---------- Helpers ----------
    void alloc_all();
    void free_all();
    void init_weights_random();

    void set_input(const Tensor& x_host);
    void forward_pass();
    
    // Tham số d_dy_host_ptr có thể là nullptr nếu dùng pipeline v1.4
    void backward_compute_gradients(const float* d_dy_host_ptr);
    void apply_sgd_update(float effective_lr);

    void forward_encoder_to_p1();
    void forward_encoder_p1_to_p2();
    void forward_decoder_from_p2();

    // Disable copy
    GPUAutoencoder(const GPUAutoencoder&) = delete;
    GPUAutoencoder& operator=(const GPUAutoencoder&) = delete;
};
