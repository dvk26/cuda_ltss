#include "gpu_autoencoder.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <fstream> // Cần thiết cho save_weights

// ====================================================
// KERNELS
// ====================================================

// [FIX] Thêm Kernel tính MSE Loss còn thiếu
__global__ void mse_loss_kernel(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ dY,
    float* __restrict__ loss_out,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float s_loss[]; 
    int tid = threadIdx.x;
    s_loss[tid] = 0.0f;

    if (idx < total_elements) {
        float p = pred[idx];
        float t = target[idx];
        float diff = p - t;
        // Gradient dY = 2/N * (P - T)
        dY[idx] = (2.0f * diff) / (float)total_elements;
        // Loss accumulation
        s_loss[tid] = diff * diff;
    }
    __syncthreads();

    // Reduction loss trong block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_loss[tid] += s_loss[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(loss_out, s_loss[0]);
    }
}

// ====================================================
// OPTIMIZED KERNELS (SHARED MEMORY)
// ====================================================

#define TILE_W 16
#define K_SIZE 3
#define HALO_R 1 // (K_SIZE - 1) / 2
#define SM_W (TILE_W + 2 * HALO_R) // 16 + 2 = 18

// ---- 1. Tiled Conv2D Forward ----
// Sử dụng Shared Memory để cache Input X
__global__ void conv2d_forward_tiled_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int Cin, int H, int W, int Cout)
{
    // Shared memory lưu patch đầu vào (18x18)
    __shared__ float s_x[SM_W][SM_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Tọa độ đầu ra của block này
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // [FIX] CUDA không có blockIdx.w. Ta phải map Z-axis = N * Cout
    int total_z = blockIdx.z;
    int c_out = total_z % Cout; // Channel out
    int n     = total_z / Cout; // Batch index n

    int h_out = by * TILE_W + ty;
    int w_out = bx * TILE_W + tx;

    // Accumulator cho kết quả
    float sum = 0.0f;
    if (h_out < H && w_out < W) {
        sum = b[c_out];
    }

    // Loop qua từng channel input
    for (int c = 0; c < Cin; ++c) {
        // 1. Load Input Tile vào Shared Memory
        int h_base = by * TILE_W - HALO_R;
        int w_base = bx * TILE_W - HALO_R;

        int tid = ty * TILE_W + tx; // 0..255
        
        // Load tile strategy: load tuyến tính
        for (int i = tid; i < SM_W * SM_W; i += (TILE_W * TILE_W)) {
            int r = i / SM_W;
            int c_sm = i % SM_W;
            
            int h_in = h_base + r;
            int w_in = w_base + c_sm;

            float val = 0.f;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                val = x[((n * Cin + c) * H + h_in) * W + w_in];
            }
            s_x[r][c_sm] = val;
        }

        __syncthreads();

        // 2. Compute Convolution cho output pixel hiện tại
        if (h_out < H && w_out < W) {
            for (int kh = 0; kh < K_SIZE; ++kh) {
                for (int kw = 0; kw < K_SIZE; ++kw) {
                    // Weight index
                    int w_idx = ((c_out * Cin + c) * K_SIZE + kh) * K_SIZE + kw;
                    // Shared mem index (ty + kh, tx + kw) do padding halo
                    float val = s_x[ty + kh][tx + kw];
                    sum += val * w[w_idx];
                }
            }
        }
        __syncthreads();
    }

    if (h_out < H && w_out < W) {
        int idx = ((n * Cout + c_out) * H + h_out) * W + w_out;
        y[idx] = sum;
    }
}

// ---- 2. Conv2D Backward DATA (dX) - NO ATOMICS ----
__global__ void conv2d_backward_data_tiled_kernel(
    const float* __restrict__ w,
    const float* __restrict__ dY,
    float* __restrict__ dX,
    int N, int Cin, int H, int W, int Cout)
{
    // Tương tự forward, dùng Shared Memory để cache dY
    __shared__ float s_dy[SM_W][SM_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // [FIX] Map Z-axis = N * Cin (vì output là dX có Cin channels)
    int total_z = blockIdx.z;
    int c_in = total_z % Cin; 
    int bz   = total_z / Cin; // batch index

    int h_in = by * TILE_W + ty;
    int w_in = bx * TILE_W + tx;

    float sum_dx = 0.0f;

    for (int c_out = 0; c_out < Cout; ++c_out) {
        // Load dY Tile
        int h_base = by * TILE_W - HALO_R;
        int w_base = bx * TILE_W - HALO_R;
        int tid = ty * TILE_W + tx;

        for (int i = tid; i < SM_W * SM_W; i += (TILE_W * TILE_W)) {
            int r = i / SM_W;
            int c_sm = i % SM_W;
            int h_dy = h_base + r;
            int w_dy = w_base + c_sm;

            float val = 0.f;
            if (h_dy >= 0 && h_dy < H && w_dy >= 0 && w_dy < W) {
                val = dY[((bz * Cout + c_out) * H + h_dy) * W + w_dy];
            }
            s_dy[r][c_sm] = val;
        }
        __syncthreads();

        // Convolution "Gather"
        if (h_in < H && w_in < W) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    int s_r = ty + 2 - kh;
                    int s_c = tx + 2 - kw;
                    
                    // Weight: [c_out, c_in, kh, kw]
                    int w_idx = ((c_out * Cin + c_in) * 3 + kh) * 3 + kw;
                    
                    sum_dx += s_dy[s_r][s_c] * w[w_idx];
                }
            }
        }
        __syncthreads();
    }

    if (h_in < H && w_in < W) {
        int idx = ((bz * Cin + c_in) * H + h_in) * W + w_in;
        dX[idx] = sum_dx;
    }
}

// ---- 3. Conv2D Backward FILTER (gW, gb) ----
__global__ void conv2d_backward_filter_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dY,
    float* __restrict__ gW,
    float* __restrict__ gb,
    int N, int Cin, int H, int W, int Cout)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * H * W;
    if (idx >= total) return;

    int w_out = idx % W;
    int tmp = idx / W;
    int h_out = tmp % H;
    tmp /= H;
    int c_out = tmp % Cout;
    int n = tmp / Cout;

    float grad = dY[idx];
    atomicAdd(&gb[c_out], grad);

    for (int c = 0; c < Cin; ++c) {
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                int ih = h_out + kh - 1;
                int iw = w_out + kw - 1;
                
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float val_x = x[((n * Cin + c) * H + ih) * W + iw];
                    int w_idx = ((c_out * Cin + c) * 3 + kh) * 3 + kw;
                    atomicAdd(&gW[w_idx], grad * val_x);
                }
            }
        }
    }
}

// ---- 3. ReLU forward / backward ----
__global__ void relu_forward_kernel(float* x, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float v = x[idx];
    x[idx] = v > 0.f ? v : 0.f;
}

__global__ void relu_backward_kernel(
    const float* __restrict__ y,
    const float* __restrict__ dY,
    float* __restrict__ dX,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    dX[idx] = (y[idx] > 0.f) ? dY[idx] : 0.f;
}

// ---- 4. MaxPool2x2 forward + backward ----
__global__ void maxpool2x2_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int* __restrict__ mask,
    int N, int C, int H, int W)
{
    int Hout = H / 2;
    int Wout = W / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Hout * Wout;
    if (idx >= total) return;

    int w_out = idx % Wout;
    int tmp = idx / Wout;
    int h_out = tmp % Hout;
    tmp /= Hout;
    int c = tmp % C;
    int n = tmp / C;

    int h0 = h_out * 2;
    int w0 = w_out * 2;

    float best = -1e30f;
    int bestk = 0;

    for (int kh = 0; kh < 2; ++kh) {
        for (int kw = 0; kw < 2; ++kw) {
            int x_idx = ((n * C + c) * H + h0 + kh) * W + w0 + kw;
            if (x[x_idx] > best) {
                best = x[x_idx];
                bestk = kh * 2 + kw;
            }
        }
    }

    int y_idx = idx;
    y[y_idx] = best;
    mask[y_idx] = bestk;
}

__global__ void maxpool2x2_backward_kernel(
    const float* __restrict__ dY,
    const int* __restrict__ mask,
    float* __restrict__ dX,
    int N, int C, int H, int W)
{
    int Hout = H / 2;
    int Wout = W / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Hout * Wout;
    if (idx >= total) return;

    int w_out = idx % Wout;
    int tmp = idx / Wout;
    int h_out = tmp % Hout;
    tmp /= Hout;
    int c = tmp % C;
    int n = tmp / C;

    int m = mask[idx];
    int kh = m / 2;
    int kw = m % 2;

    int ih = h_out * 2 + kh;
    int iw = w_out * 2 + kw;

    int x_idx = ((n * C + c) * H + ih) * W + iw;
    atomicAdd(&dX[x_idx], dY[idx]);
}

// ---- 5. Upsample2x (nearest-neighbor) forward/backward ----
__global__ void upsample2x_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W)
{
    int Hout = H * 2;
    int Wout = W * 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Hout * Wout;
    if (idx >= total) return;

    int w_out = idx % Wout;
    int tmp = idx / Wout;
    int h_out = tmp % Hout;
    tmp /= Hout;
    int c = tmp % C;
    int n = tmp / C;

    int ih = h_out / 2;
    int iw = w_out / 2;

    int x_idx = ((n * C + c) * H + ih) * W + iw;
    y[idx] = x[x_idx];
}

__global__ void upsample2x_backward_kernel(
    const float* __restrict__ dY,
    float* __restrict__ dX,
    int N, int C, int H, int W)
{
    int Hout = H * 2;
    int Wout = W * 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w_in = idx % W;
    int tmp = idx / W;
    int h_in = tmp % H;
    tmp /= H;
    int c = tmp % C;
    int n = tmp / C;

    float sum = 0.f;
    for (int kh = 0; kh < 2; ++kh) {
        for (int kw = 0; kw < 2; ++kw) {
            int y_idx = ((n * C + c) * Hout + h_in * 2 + kh) * Wout + w_in * 2 + kw;
            sum += dY[y_idx];
        }
    }
    dX[idx] = sum;
}

// ---- 6. Zero buffer ----
__global__ void zero_kernel(float* x, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    x[idx] = 0.f;
}

// ---- 7. SGD update ----
__global__ void sgd_update_kernel(
    float* __restrict__ w,
    const float* __restrict__ gw,
    float lr,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    w[idx] -= lr * gw[idx];
}

// ====================================================
// GPUAutoencoder implementation
// ====================================================

GPUAutoencoder::GPUAutoencoder(int batch_size, int H, int W)
    : N_(batch_size), H_(H), W_(W)
{
    H1_ = H_ / 2; W1_ = W_ / 2;
    H2_ = H1_ / 2; W2_ = W1_ / 2;
    alloc_all();
    init_weights_random();
}

GPUAutoencoder::~GPUAutoencoder()
{
    free_all();
}

void GPUAutoencoder::alloc_all()
{
    size_t sz_w1 = 256 * 3 * 3 * 3;
    size_t sz_w2 = 128 * 256 * 3 * 3;
    size_t sz_w3 = 128 * 128 * 3 * 3;
    size_t sz_w4 = 256 * 128 * 3 * 3;
    size_t sz_w5 = 3   * 256 * 3 * 3;

    CUDA_CHECK(cudaMalloc(&d_w1_, sz_w1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1_, 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gw1_, sz_w1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gb1_, 256 * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_w2_, sz_w2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2_, 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gw2_, sz_w2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gb2_, 128 * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_w3_, sz_w3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b3_, 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gw3_, sz_w3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gb3_, 128 * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_w4_, sz_w4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b4_, 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gw4_, sz_w4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gb4_, 256 * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_w5_, sz_w5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b5_, 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gw5_, sz_w5 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gb5_, 3 * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_x_,  nchw_size(N_, 3,   H_,  W_)   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c1_, nchw_size(N_, 256, H_,  W_)   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r1_, nchw_size(N_, 256, H_,  W_)   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p1_, nchw_size(N_, 256, H1_, W1_)  * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_c2_, nchw_size(N_, 128, H1_, W1_) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r2_, nchw_size(N_, 128, H1_, W1_) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_p2_, nchw_size(N_, 128, H2_, W2_) * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_c3_, nchw_size(N_, 128, H2_, W2_) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r3_, nchw_size(N_, 128, H2_, W2_) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u1_, nchw_size(N_, 128, H1_, W1_) * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_c4_, nchw_size(N_, 256, H1_, W1_) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_r4_, nchw_size(N_, 256, H1_, W1_) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u2_, nchw_size(N_, 256, H_,  W_)  * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_c5_, nchw_size(N_, 3,   H_,  W_)   * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_mask1_, nchw_size(N_, 256, H1_, W1_) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_mask2_, nchw_size(N_, 128, H2_, W2_) * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_dc5_, nchw_size(N_, 3,   H_,  W_)   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_du2_, nchw_size(N_, 256, H_,  W_)   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dr4_, nchw_size(N_, 256, H1_, W1_)  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dc4_, nchw_size(N_, 256, H1_, W1_)  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_du1_, nchw_size(N_, 128, H1_, W1_)  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dr3_, nchw_size(N_, 128, H2_, W2_)  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dc3_, nchw_size(N_, 128, H2_, W2_)  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dp2_, nchw_size(N_, 128, H2_, W2_)  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dr2_, nchw_size(N_, 128, H1_, W1_)  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dc2_, nchw_size(N_, 128, H1_, W1_)  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dp1_, nchw_size(N_, 256, H1_, W1_)  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dr1_, nchw_size(N_, 256, H_,  W_)   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dc1_, nchw_size(N_, 256, H_,  W_)   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dx_,  nchw_size(N_, 3,   H_,  W_)   * sizeof(float)));

    // [FIX] Cấp phát 2 biến quan trọng bị thiếu
    CUDA_CHECK(cudaMalloc(&d_dy_, nchw_size(N_, 3, H_, W_) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss_accum_, sizeof(float)));
}

void GPUAutoencoder::free_all()
{
    auto safe_free = [](auto*& ptr) { if (ptr) { cudaFree(ptr); ptr = nullptr; } };

    safe_free(d_w1_); safe_free(d_b1_); safe_free(d_gw1_); safe_free(d_gb1_);
    safe_free(d_w2_); safe_free(d_b2_); safe_free(d_gw2_); safe_free(d_gb2_);
    safe_free(d_w3_); safe_free(d_b3_); safe_free(d_gw3_); safe_free(d_gb3_);
    safe_free(d_w4_); safe_free(d_b4_); safe_free(d_gw4_); safe_free(d_gb4_);
    safe_free(d_w5_); safe_free(d_b5_); safe_free(d_gw5_); safe_free(d_gb5_);

    safe_free(d_x_);  safe_free(d_c1_); safe_free(d_r1_); safe_free(d_p1_);
    safe_free(d_c2_); safe_free(d_r2_); safe_free(d_p2_);
    safe_free(d_c3_); safe_free(d_r3_); safe_free(d_u1_);
    safe_free(d_c4_); safe_free(d_r4_); safe_free(d_u2_);
    safe_free(d_c5_);

    safe_free(d_mask1_); safe_free(d_mask2_);

    safe_free(d_dc5_); safe_free(d_du2_); safe_free(d_dr4_); safe_free(d_dc4_);
    safe_free(d_du1_); safe_free(d_dr3_); safe_free(d_dc3_);
    safe_free(d_dp2_); safe_free(d_dr2_); safe_free(d_dc2_);
    safe_free(d_dp1_); safe_free(d_dr1_); safe_free(d_dc1_);
    safe_free(d_dx_);
    
    // [FIX] Free 2 biến mới
    safe_free(d_dy_);
    safe_free(d_loss_accum_);
}

void GPUAutoencoder::init_weights_random()
{
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    auto init_kaiming = [&](float* d_ptr, int k_size, int cin, int cout) {
        float std_val = std::sqrt(2.0f / (k_size * k_size * cin));
        std::vector<float> h_data(cout * cin * k_size * k_size);
        for (auto& v : h_data) v = dist(rng) * std_val;
        CUDA_CHECK(cudaMemcpy(d_ptr, h_data.data(), h_data.size() * sizeof(float), cudaMemcpyHostToDevice));
    };

    auto init_zero = [&](float* d_ptr, size_t n) {
        std::vector<float> h_data(n, 0.f);
        CUDA_CHECK(cudaMemcpy(d_ptr, h_data.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    };

    init_kaiming(d_w1_, 3, 3, 256);   init_zero(d_b1_, 256);
    init_kaiming(d_w2_, 3, 256, 128); init_zero(d_b2_, 128);
    init_kaiming(d_w3_, 3, 128, 128); init_zero(d_b3_, 128);
    init_kaiming(d_w4_, 3, 128, 256); init_zero(d_b4_, 256);
    init_kaiming(d_w5_, 3, 256, 3);   init_zero(d_b5_, 3);
}

void GPUAutoencoder::set_input(const Tensor& x_host)
{
    size_t sz = nchw_size(N_, 3, H_, W_);
    CUDA_CHECK(cudaMemcpy(d_x_, x_host.raw().data(), sz * sizeof(float), cudaMemcpyHostToDevice));
}

void launch_conv2d_tiled(
    const float* x, const float* w, const float* b, float* y,
    int N, int Cin, int H, int W, int Cout)
{
    dim3 block(16, 16);
    // [FIX] CUDA chỉ hỗ trợ 3 chiều Grid. Map Z-axis = N * Cout
    dim3 grid(
        (W + 15) / 16, 
        (H + 15) / 16, 
        N * Cout
    );
    conv2d_forward_tiled_kernel<<<grid, block>>>(x, w, b, y, N, Cin, H, W, Cout);
    CUDA_CHECK(cudaGetLastError());
}

void GPUAutoencoder::forward_pass()
{
    const int BS = 256;
    dim3 block(BS);

    // Conv1
    {
        launch_conv2d_tiled(d_x_, d_w1_, d_b1_, d_c1_, N_, 3, H_, W_, 256);
        CUDA_CHECK(cudaGetLastError());
    }

    // ReLU1: c1 -> r1
    {
        int total = N_ * 256 * H_ * W_;
        dim3 grid((total + BS - 1) / BS);
        relu_forward_kernel<<<grid, block>>>(d_c1_, total);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(d_r1_, d_c1_, total * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Pool1: r1 -> p1
    {
        int total = N_ * 256 * H1_ * W1_;
        dim3 grid((total + BS - 1) / BS);
        maxpool2x2_forward_kernel<<<grid, block>>>(
            d_r1_, d_p1_, d_mask1_, N_, 256, H_, W_);
        CUDA_CHECK(cudaGetLastError());
    }

    // Conv2: 256->128
    {
        launch_conv2d_tiled(d_p1_, d_w2_, d_b2_, d_c2_, N_, 256, H1_, W1_, 128);
        CUDA_CHECK(cudaGetLastError());
    }

    // ReLU2
    {
        int total = N_ * 128 * H1_ * W1_;
        dim3 grid((total + BS - 1) / BS);
        relu_forward_kernel<<<grid, block>>>(d_c2_, total);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(d_r2_, d_c2_, total * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Pool2: r2 -> p2 (latent)
    {
        int total = N_ * 128 * H2_ * W2_;
        dim3 grid((total + BS - 1) / BS);
        maxpool2x2_forward_kernel<<<grid, block>>>(
            d_r2_, d_p2_, d_mask2_, N_, 128, H1_, W1_);
        CUDA_CHECK(cudaGetLastError());
    }

    // ---- Decoder ----

    // Conv3: 128->128
    {
        launch_conv2d_tiled(d_p2_, d_w3_, d_b3_, d_c3_, N_, 128, H2_, W2_, 128);
        CUDA_CHECK(cudaGetLastError());
    }

    // ReLU3
    {
        int total = N_ * 128 * H2_ * W2_;
        dim3 grid((total + BS - 1) / BS);
        relu_forward_kernel<<<grid, block>>>(d_c3_, total);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(d_r3_, d_c3_, total * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Upsample1: [N,128,H2,W2] -> [N,128,H1,W1]
    {
        int total = N_ * 128 * H1_ * W1_;
        dim3 grid((total + BS - 1) / BS);
        upsample2x_forward_kernel<<<grid, block>>>(
            d_r3_, d_u1_, N_, 128, H2_, W2_);
        CUDA_CHECK(cudaGetLastError());
    }

    // Conv4: 128->256
    {
        launch_conv2d_tiled(d_u1_, d_w4_, d_b4_, d_c4_, N_, 128, H1_, W1_, 256);
        CUDA_CHECK(cudaGetLastError());
    }

    // ReLU4
    {
        int total = N_ * 256 * H1_ * W1_;
        dim3 grid((total + BS - 1) / BS);
        relu_forward_kernel<<<grid, block>>>(d_c4_, total);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(d_r4_, d_c4_, total * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Upsample2: [N,256,H1,W1] -> [N,256,H,W]
    {
        int total = N_ * 256 * H_ * W_;
        dim3 grid((total + BS - 1) / BS);
        upsample2x_forward_kernel<<<grid, block>>>(
            d_r4_, d_u2_, N_, 256, H1_, W1_);
        CUDA_CHECK(cudaGetLastError());
    }

    // Conv5: 256->3 (output)
    {
        launch_conv2d_tiled(d_u2_, d_w5_, d_b5_, d_c5_, N_, 256, H_, W_, 3);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

Tensor GPUAutoencoder::encode(const Tensor& x_host)
{
    if (x_host.N() != N_ || x_host.C() != 3 || x_host.H() != H_ || x_host.W() != W_) {
        throw std::runtime_error("Input shape mismatch in encode");
    }

    set_input(x_host);

    // Forward encoder only
    const int BS = 256;
    dim3 block(BS);

    // Conv1 (Optimized)
    launch_conv2d_tiled(d_x_, d_w1_, d_b1_, d_c1_, N_, 3, H_, W_, 256);

    // ReLU1
    {
        int total = N_ * 256 * H_ * W_;
        dim3 grid((total + BS - 1) / BS);
        relu_forward_kernel<<<grid, block>>>(d_c1_, total);
        CUDA_CHECK(cudaMemcpy(d_r1_, d_c1_, total * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Pool1
    {
        int total = N_ * 256 * H1_ * W1_;
        dim3 grid((total + BS - 1) / BS);
        maxpool2x2_forward_kernel<<<grid, block>>>(d_r1_, d_p1_, d_mask1_, N_, 256, H_, W_);
    }

    // Conv2 (Optimized)
    launch_conv2d_tiled(d_p1_, d_w2_, d_b2_, d_c2_, N_, 256, H1_, W1_, 128);

    // ReLU2
    {
        int total = N_ * 128 * H1_ * W1_;
        dim3 grid((total + BS - 1) / BS);
        relu_forward_kernel<<<grid, block>>>(d_c2_, total);
        CUDA_CHECK(cudaMemcpy(d_r2_, d_c2_, total * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Pool2 (Latent)
    {
        int total = N_ * 128 * H2_ * W2_;
        dim3 grid((total + BS - 1) / BS);
        maxpool2x2_forward_kernel<<<grid, block>>>(d_r2_, d_p2_, d_mask2_, N_, 128, H1_, W1_);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy latent to host
    Tensor latent(N_, 128, H2_, W2_);
    size_t sz = nchw_size(N_, 128, H2_, W2_) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(latent.raw().data(), d_p2_, sz, cudaMemcpyDeviceToHost));

    return latent;
}

Tensor GPUAutoencoder::decode(const Tensor& z_host)
{
    if (z_host.N() != N_ || z_host.C() != 128 || z_host.H() != H2_ || z_host.W() != W2_) {
        throw std::runtime_error("Latent shape mismatch in decode");
    }

    // Copy latent to GPU
    size_t sz = nchw_size(N_, 128, H2_, W2_) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_p2_, z_host.raw().data(), sz, cudaMemcpyHostToDevice));

    const int BS = 256;
    dim3 block(BS);

    // Conv3 (Optimized)
    launch_conv2d_tiled(d_p2_, d_w3_, d_b3_, d_c3_, N_, 128, H2_, W2_, 128);

    // ReLU3
    {
        int total = N_ * 128 * H2_ * W2_;
        dim3 grid((total + BS - 1) / BS);
        relu_forward_kernel<<<grid, block>>>(d_c3_, total);
        CUDA_CHECK(cudaMemcpy(d_r3_, d_c3_, total * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Upsample1
    {
        int total = N_ * 128 * H1_ * W1_;
        dim3 grid((total + BS - 1) / BS);
        upsample2x_forward_kernel<<<grid, block>>>(d_r3_, d_u1_, N_, 128, H2_, W2_);
    }

    // Conv4 (Optimized)
    launch_conv2d_tiled(d_u1_, d_w4_, d_b4_, d_c4_, N_, 128, H1_, W1_, 256);

    // ReLU4
    {
        int total = N_ * 256 * H1_ * W1_;
        dim3 grid((total + BS - 1) / BS);
        relu_forward_kernel<<<grid, block>>>(d_c4_, total);
        CUDA_CHECK(cudaMemcpy(d_r4_, d_c4_, total * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Upsample2
    {
        int total = N_ * 256 * H_ * W_;
        dim3 grid((total + BS - 1) / BS);
        upsample2x_forward_kernel<<<grid, block>>>(d_r4_, d_u2_, N_, 256, H1_, W1_);
    }

    // Conv5 (Optimized)
    launch_conv2d_tiled(d_u2_, d_w5_, d_b5_, d_c5_, N_, 256, H_, W_, 3);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output to host
    Tensor output(N_, 3, H_, W_);
    sz = nchw_size(N_, 3, H_, W_) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(output.raw().data(), d_c5_, sz, cudaMemcpyDeviceToHost));

    return output;
}

Tensor GPUAutoencoder::forward(const Tensor& x_host)
{
    if (x_host.N() != N_ || x_host.C() != 3 || x_host.H() != H_ || x_host.W() != W_) {
        throw std::runtime_error("Input shape mismatch");
    }

    set_input(x_host);
    forward_pass();

    Tensor output(N_, 3, H_, W_);
    size_t sz = nchw_size(N_, 3, H_, W_) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(output.raw().data(), d_c5_, sz, cudaMemcpyDeviceToHost));

    return output;
}

// Helper cho Backward
void launch_conv2d_backward(
    const float* x, const float* w, const float* dY, 
    float* dX, float* gW, float* gb,
    int N, int Cin, int H, int W, int Cout)
{
    // 1. Tính dX (Dùng Tiled Kernel, không atomic)
    // Grid structure: (GridW, GridH, Batch, Cin) <-- Output là Cin
    dim3 block(16, 16);
    // [FIX] Map Z-axis = N * Cin
    dim3 grid_data(
        (W + 15) / 16,
        (H + 15) / 16,
        N * Cin 
    );
    conv2d_backward_data_tiled_kernel<<<grid_data, block>>>(w, dY, dX, N, Cin, H, W, Cout);
    CUDA_CHECK(cudaGetLastError());

    // 2. Tính gW, gb (Dùng Kernel Atomic cũ nhưng riêng biệt)
    const int BS = 256;
    dim3 block_filter(BS);
    dim3 grid_filter((N * Cout * H * W + BS - 1) / BS);
    conv2d_backward_filter_kernel<<<grid_filter, block_filter>>>(x, dY, gW, gb, N, Cin, H, W, Cout);
    CUDA_CHECK(cudaGetLastError());
}

// [FIX] Sửa logic tham số truyền vào để xử lý trường hợp nullptr
void GPUAutoencoder::backward_pass(const float* external_grad, float lr)
{
    const int BS = 256;
    dim3 block(BS);

    // dC5 = dY
    {
        int total = N_ * 3 * H_ * W_;
        if (external_grad != nullptr) {
            // Nếu có gradient từ ngoài (vd: từ backward_and_update) -> Copy vào
            CUDA_CHECK(cudaMemcpy(d_dc5_, external_grad, total * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            // Nếu null (vd: từ train_step) -> Copy từ buffer nội bộ d_dy_
            CUDA_CHECK(cudaMemcpy(d_dc5_, d_dy_, total * sizeof(float), cudaMemcpyDeviceToDevice));
        }
    }

    // Zero gradient buffers
    auto zero_buf = [&](float* p, size_t n) {
        dim3 grid((n + BS - 1) / BS);
        zero_kernel<<<grid, block>>>(p, n);
        CUDA_CHECK(cudaGetLastError());
    };

    zero_buf(d_gw1_, 256 * 3 * 3 * 3);
    zero_buf(d_gb1_, 256);
    zero_buf(d_gw2_, 128 * 256 * 3 * 3);
    zero_buf(d_gb2_, 128);
    zero_buf(d_gw3_, 128 * 128 * 3 * 3);
    zero_buf(d_gb3_, 128);
    zero_buf(d_gw4_, 256 * 128 * 3 * 3);
    zero_buf(d_gb4_, 256);
    zero_buf(d_gw5_, 3 * 256 * 3 * 3);
    zero_buf(d_gb5_, 3);

    zero_buf(d_du2_, nchw_size(N_,256,H_,W_));
    zero_buf(d_dr4_, nchw_size(N_,256,H1_,W1_));
    zero_buf(d_dc4_, nchw_size(N_,256,H1_,W1_));
    zero_buf(d_du1_, nchw_size(N_,128,H1_,W1_));
    zero_buf(d_dr3_, nchw_size(N_,128,H2_,W2_));
    zero_buf(d_dc3_, nchw_size(N_,128,H2_,W2_));
    zero_buf(d_dp2_, nchw_size(N_,128,H2_,W2_));
    zero_buf(d_dr2_, nchw_size(N_,128,H1_,W1_));
    zero_buf(d_dc2_, nchw_size(N_,128,H1_,W1_));
    zero_buf(d_dp1_, nchw_size(N_,256,H1_,W1_));
    zero_buf(d_dr1_, nchw_size(N_,256,H_,W_));
    zero_buf(d_dc1_, nchw_size(N_,256,H_,W_));
    zero_buf(d_dx_,  nchw_size(N_,3,H_,W_));

    // Backward qua Conv5
    {
        launch_conv2d_backward(d_u2_, d_w5_, d_dc5_, d_du2_, d_gw5_, d_gb5_, N_, 256, H_, W_, 3);
        CUDA_CHECK(cudaGetLastError());
    }

    // Up2 backward
    {
        int total = N_ * 256 * H1_ * W1_;
        dim3 grid((total + BS - 1) / BS);
        upsample2x_backward_kernel<<<grid, block>>>(
            d_du2_, d_dr4_, N_, 256, H1_, W1_);
        CUDA_CHECK(cudaGetLastError());
    }

    // ReLU4 backward
    {
        int total = N_ * 256 * H1_ * W1_;
        dim3 grid((total + BS - 1) / BS);
        relu_backward_kernel<<<grid, block>>>(
            d_r4_, d_dr4_, d_dc4_, total);
        CUDA_CHECK(cudaGetLastError());
    }

    // Conv4 backward
    {
        launch_conv2d_backward(d_u1_, d_w4_, d_dc4_, d_du1_, d_gw4_, d_gb4_, N_, 128, H1_, W1_, 256);
        CUDA_CHECK(cudaGetLastError());
    }

    // Up1 backward
    {
        int total = N_ * 128 * H2_ * W2_;
        dim3 grid((total + BS - 1) / BS);
        upsample2x_backward_kernel<<<grid, block>>>(
            d_du1_, d_dr3_, N_, 128, H2_, W2_);
        CUDA_CHECK(cudaGetLastError());
    }

    // ReLU3 backward
    {
        int total = N_ * 128 * H2_ * W2_;
        dim3 grid((total + BS - 1) / BS);
        relu_backward_kernel<<<grid, block>>>(
            d_r3_, d_dr3_, d_dc3_, total);
        CUDA_CHECK(cudaGetLastError());
    }

    // Conv3 backward
    {
        launch_conv2d_backward(d_p2_, d_w3_, d_dc3_, d_dp2_, d_gw3_, d_gb3_, N_, 128, H2_, W2_, 128);
        CUDA_CHECK(cudaGetLastError());
    }

    // Pool2 backward
    {
        int total = N_ * 128 * H2_ * W2_;
        dim3 grid((total + BS - 1) / BS);
        maxpool2x2_backward_kernel<<<grid, block>>>(
            d_dp2_, d_mask2_, d_dr2_, N_, 128, H1_, W1_);
        CUDA_CHECK(cudaGetLastError());
    }

    // ReLU2 backward
    {
        int total = N_ * 128 * H1_ * W1_;
        dim3 grid((total + BS - 1) / BS);
        relu_backward_kernel<<<grid, block>>>(
            d_r2_, d_dr2_, d_dc2_, total);
        CUDA_CHECK(cudaGetLastError());
    }

    // Conv2 backward
    {
        launch_conv2d_backward(d_p1_, d_w2_, d_dc2_, d_dp1_, d_gw2_, d_gb2_, N_, 256, H1_, W1_, 128);
        CUDA_CHECK(cudaGetLastError());
    }

    // Pool1 backward
    {
        int total = N_ * 256 * H1_ * W1_;
        dim3 grid((total + BS - 1) / BS);
        maxpool2x2_backward_kernel<<<grid, block>>>(
            d_dp1_, d_mask1_, d_dr1_, N_, 256, H_, W_);
        CUDA_CHECK(cudaGetLastError());
    }

    // ReLU1 backward
    {
        int total = N_ * 256 * H_ * W_;
        dim3 grid((total + BS - 1) / BS);
        relu_backward_kernel<<<grid, block>>>(
            d_r1_, d_dr1_, d_dc1_, total);
        CUDA_CHECK(cudaGetLastError());
    }

    // Conv1 backward
    {
        launch_conv2d_backward(d_x_, d_w1_, d_dc1_, d_dx_, d_gw1_, d_gb1_, N_, 3, H_, W_, 256);
        CUDA_CHECK(cudaGetLastError());
    }

    // SGD update
    auto sgd = [&](float* w, float* gw, int n) {
        dim3 grid((n + BS - 1) / BS);
        sgd_update_kernel<<<grid, block>>>(w, gw, lr, n);
        CUDA_CHECK(cudaGetLastError());
    };

    sgd(d_w1_, d_gw1_, 256 * 3 * 3 * 3);
    sgd(d_b1_, d_gb1_, 256);
    sgd(d_w2_, d_gw2_, 128 * 256 * 3 * 3);
    sgd(d_b2_, d_gb2_, 128);
    sgd(d_w3_, d_gw3_, 128 * 128 * 3 * 3);
    sgd(d_b3_, d_gb3_, 128);
    sgd(d_w4_, d_gw4_, 256 * 128 * 3 * 3);
    sgd(d_b4_, d_gb4_, 256);
    sgd(d_w5_, d_gw5_, 3 * 256 * 3 * 3);
    sgd(d_b5_, d_gb5_, 3);

    CUDA_CHECK(cudaDeviceSynchronize());
}

// [FIX] Cập nhật hàm train_step để dùng mse_loss_kernel
float GPUAutoencoder::train_step(const Tensor& x_host, float lr)
{
    set_input(x_host);
    forward_pass();

    // Reset loss accumulation
    CUDA_CHECK(cudaMemset(d_loss_accum_, 0, sizeof(float)));

    int total = N_ * 3 * H_ * W_;
    const int BS = 256;
    
    // Gọi kernel MSE để tính Loss và Gradient dY
    mse_loss_kernel<<<(total + BS - 1) / BS, BS, BS * sizeof(float)>>>(
        d_c5_, d_x_, d_dy_, d_loss_accum_, total
    );

    // Truyền nullptr để backward_pass biết dùng d_dy_ nội bộ
    backward_pass(nullptr, lr);

    float total_loss = 0.0f;
    CUDA_CHECK(cudaMemcpy(&total_loss, d_loss_accum_, sizeof(float), cudaMemcpyDeviceToHost));
    
    return total_loss / (float)total;
}

float GPUAutoencoder::compute_loss(const Tensor& x_host)
{
    set_input(x_host);
    forward_pass();

    CUDA_CHECK(cudaMemset(d_loss_accum_, 0, sizeof(float)));
    int total = N_ * 3 * H_ * W_;
    const int BS = 256;
    mse_loss_kernel<<<(total + BS - 1)/BS, BS, BS * sizeof(float)>>>(d_c5_, d_x_, d_dy_, d_loss_accum_, total);

    float total_loss;
    CUDA_CHECK(cudaMemcpy(&total_loss, d_loss_accum_, sizeof(float), cudaMemcpyDeviceToHost));
    return total_loss / (float)total;
}