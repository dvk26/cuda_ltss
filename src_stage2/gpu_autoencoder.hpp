#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <random>
#include "tensor.hpp"
#include "autoencoder.hpp" 

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
// GPU Autoencoder
// Enc: 3->256 -> pool2 -> 256->128 -> pool2
// Dec: 128->128 -> up2 -> 128->256 -> up2 -> 256->3
// =====================

class GPUAutoencoder {
public:
    GPUAutoencoder(int batch_size, int H = 32, int W = 32);
    ~GPUAutoencoder();

    // Copy weight từ Autoencoder CPU sang GPU
    void copy_weights_from_cpu(const Autoencoder& cpu);
    // Copy ngược lại (để so kết quả / lưu)
    void copy_weights_to_cpu(Autoencoder& cpu) const;

    // Forward-only: x_host -> y_host (không backprop)
    void forward(const Tensor& x_host, Tensor& y_host);

    // Train 1 batch: x_host là input, lr là learning rate
    // Trả về loss (MSE)
    float train_batch(const Tensor& x_host, float lr);
    // Getter const
    const Conv2D& c1() const { return c1_; }
    const Conv2D& c2() const { return c2_; }
    const Conv2D& c3() const { return c3_; }
    const Conv2D& c4() const { return c4_; }
    const Conv2D& c5() const { return c5_; }

    // Getter non-const (dùng để nhận weight từ GPU copy về)
    Conv2D& c1() { return c1_; }
    Conv2D& c2() { return c2_; }
    Conv2D& c3() { return c3_; }
    Conv2D& c4() { return c4_; }
    Conv2D& c5() { return c5_; }


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

    // ---------- MSE loss ----------
    float* d_target_; // [N,3,32,32]
    float* d_dy_;     // dL/dy (same shape)
    float* d_loss_accum_; // 1 float trên device

    // ---------- Helpers ----------
    void alloc_all();
    void free_all();
    void init_weights_random();

    float forward_internal();  // dùng d_x_ -> d_c5_, không copy từ host
    float backward_internal(float lr); // backprop từ d_dy_

    // Không cho copy
    GPUAutoencoder(const GPUAutoencoder&) = delete;
    GPUAutoencoder& operator=(const GPUAutoencoder&) = delete;



};
