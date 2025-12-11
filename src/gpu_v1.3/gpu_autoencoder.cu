#include "gpu_autoencoder.hpp"
#include <iostream>
#include <cmath>

// ====================================================
// KERNELS
// ====================================================


// ====================================================
// OPTIMIZED KERNELS (SHARED MEMORY)
// ====================================================

#define TILE_W 16
#define K_SIZE 3
#define HALO_R 1 
#define SM_W (TILE_W + 2 * HALO_R) // 18

// ---- 0. MSE Loss Kernel (Moved Up) ----
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
        dY[idx] = (2.0f * diff) / (float)total_elements; // Gradient
        s_loss[tid] = diff * diff; // Loss
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s_loss[tid] += s_loss[tid + stride];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(loss_out, s_loss[0]);
}

// ---- 1a. Tiled Conv2D Forward (Standard - Cho lớp cuối) ----
__global__ void conv2d_forward_tiled_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int Cin, int H, int W, int Cout)
{
    __shared__ float s_x[SM_W][SM_W];
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;
    int n = bz; int c_out = blockIdx.w;
    int h_out = by * TILE_W + ty;
    int w_out = bx * TILE_W + tx;

    float sum = 0.0f;
    if (h_out < H && w_out < W) sum = b[c_out];

    for (int c = 0; c < Cin; ++c) {
        int h_base = by * TILE_W - HALO_R;
        int w_base = bx * TILE_W - HALO_R;
        int tid = ty * TILE_W + tx;
        
        for (int i = tid; i < SM_W * SM_W; i += (TILE_W * TILE_W)) {
            int r = i / SM_W; int c_sm = i % SM_W;
            int h_in = h_base + r; int w_in = w_base + c_sm;
            float val = 0.f;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) 
                val = x[((n * Cin + c) * H + h_in) * W + w_in];
            s_x[r][c_sm] = val;
        }
        __syncthreads();

        if (h_out < H && w_out < W) {
            for (int kh = 0; kh < K_SIZE; ++kh) {
                for (int kw = 0; kw < K_SIZE; ++kw) {
                    int w_idx = ((c_out * Cin + c) * K_SIZE + kh) * K_SIZE + kw;
                    sum += s_x[ty + kh][tx + kw] * w[w_idx];
                }
            }
        }
        __syncthreads();
    }
    if (h_out < H && w_out < W) {
        y[((n * Cout + c_out) * H + h_out) * W + w_out] = sum;
    }
}

// ---- 1b. [NEW] Fused Conv2D + ReLU Forward (v1.4) ----
__global__ void conv2d_relu_forward_tiled_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int Cin, int H, int W, int Cout)
{
    // ... Copy Logic Load Shared Memory giống hệt kernel 1a ...
    __shared__ float s_x[SM_W][SM_W];
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;
    int n = bz; int c_out = blockIdx.w;
    int h_out = by * TILE_W + ty;
    int w_out = bx * TILE_W + tx;

    float sum = 0.0f;
    if (h_out < H && w_out < W) sum = b[c_out];

    for (int c = 0; c < Cin; ++c) {
        int h_base = by * TILE_W - HALO_R;
        int w_base = bx * TILE_W - HALO_R;
        int tid = ty * TILE_W + tx;
        for (int i = tid; i < SM_W * SM_W; i += (TILE_W * TILE_W)) {
            int r = i / SM_W; int c_sm = i % SM_W;
            int h_in = h_base + r; int w_in = w_base + c_sm;
            float val = 0.f;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) 
                val = x[((n * Cin + c) * H + h_in) * W + w_in];
            s_x[r][c_sm] = val;
        }
        __syncthreads();
        if (h_out < H && w_out < W) {
            for (int kh = 0; kh < K_SIZE; ++kh) {
                for (int kw = 0; kw < K_SIZE; ++kw) {
                    int w_idx = ((c_out * Cin + c) * K_SIZE + kh) * K_SIZE + kw;
                    sum += s_x[ty + kh][tx + kw] * w[w_idx];
                }
            }
        }
        __syncthreads();
    }
    
    // --- FUSION: Apply ReLU ---
    if (h_out < H && w_out < W) {
        float val = (sum > 0.f) ? sum : 0.f; 
        y[((n * Cout + c_out) * H + h_out) * W + w_out] = val;
    }
}
// ---- 2. Conv2D Backward DATA (dX) - NO ATOMICS ----
// Tính dX bằng cách gom gradient từ dY xung quanh (Gather)
// Đây thực chất là một phép Convolution của dY với Weights (được lật)
__global__ void conv2d_backward_data_tiled_kernel(
    const float* __restrict__ w,
    const float* __restrict__ dY,
    float* __restrict__ dX,
    int N, int Cin, int H, int W, int Cout)
{
    // Chúng ta tính dX[n, c_in, h, w]
    // Cần sum(dY[n, c_out, h', w'] * W[c_out, c_in, kh, kw])
    
    // Tương tự forward, dùng Shared Memory để cache dY
    __shared__ float s_dy[SM_W][SM_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z; // n
    int c_in = blockIdx.w; // c_in

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
        // dX tại (h,w) nhận đóng góp từ dY tại (h+kh-1, w+kw-1)
        // Lưu ý: index weight phải map ngược lại vì đây là correlation
        // Công thức chuẩn: dX[h,w] += dY[h+i, w+j] * W[c_out, c_in, 1-i, 1-j] (với kernel 3x3, zero centered)
        // Map sang 0..2: dY tại offset (kh-1), weight tại (2-kh)
        
        if (h_in < H && w_in < W) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    // Lấy dY từ shared mem
                    // Tại sao lại là [ty + (2-kh)][tx + (2-kw)]?
                    // Đây là do phép lật kernel trong backward pass.
                    // Tuy nhiên để đơn giản hoá tư duy:
                    // Pixel Input(h,w) ảnh hưởng đến Output(h-1, w-1) qua Weight(2,2)
                    // ... ảnh hưởng đến Output(h+1, w+1) qua Weight(0,0)
                    // Nên ta cần lấy dY(h-1..h+1). 
                    // Trong shared mem, dY đã load centered.
                    // Weight index: [c_out, c_in, kh, kw]
                    // dY offset tương ứng: (1-kh), (1-kw)
                    // Shared mem index: ty + 1 + (1-kh) = ty + 2 - kh
                    
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
// Cái này khó bỏ atomicAdd mà không dùng buffer lớn hoặc GEMM.
// Giữ lại atomic nhưng code gọn hơn để giảm register pressure.
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

// --- Giữ nguyên các kernel khác (ReLU, Pool, Upsample, Zero, SGD) ---
// ... (Copy từ code cũ vào đây nếu chưa có) ...


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
    dim3 grid(
        (W + 15) / 16, 
        (H + 15) / 16, 
        N, 
        Cout
    );
    conv2d_forward_tiled_kernel<<<grid, block>>>(x, w, b, y, N, Cin, H, W, Cout);
    CUDA_CHECK(cudaGetLastError());
}

// Wrapper cho Fused Kernel (Conv + ReLU)
void launch_conv2d_relu_tiled(
    const float* x, const float* w, const float* b, float* y,
    int N, int Cin, int H, int W, int Cout)
{
    dim3 block(16, 16);
    dim3 grid((W + 15)/16, (H + 15)/16, N, Cout);
    conv2d_relu_forward_tiled_kernel<<<grid, block>>>(x, w, b, y, N, Cin, H, W, Cout);
    CUDA_CHECK(cudaGetLastError());
}

void GPUAutoencoder::forward_pass()
{
    const int BS = 256;
    dim3 block(BS);

    // Conv1 + ReLU1
    {
        launch_conv2d_relu_tiled(d_x_, d_w1_, d_b1_, d_r1_, N_, 3, H_, W_, 256);
        // Copy d_c1_ sang d_r1_ để backward dùng (nếu code backward chưa sửa thành in-place)
        // Hoặc nếu backward code vẫn dùng d_r1_, ta cứ copy cho an toàn logic cũ
        // (Ở v1.4 tối ưu hơn nữa thì bỏ hẳn d_r1_, d_r2_ nhưng để code chạy được ngay thì giữ copy)
        CUDA_CHECK(cudaMemcpy(d_r1_, d_c1_, nchw_size(N_,256,H_,W_)*sizeof(float), cudaMemcpyDeviceToDevice));
    }

    

    // Pool1: r1 -> p1
    {
        int total = N_ * 256 * H1_ * W1_;
        dim3 grid((total + BS - 1) / BS);
        maxpool2x2_forward_kernel<<<grid, block>>>(
            d_r1_, d_p1_, d_mask1_, N_, 256, H_, W_);
        CUDA_CHECK(cudaGetLastError());
    }

    // Conv2: 256->128 + ReLU2
    {
        launch_conv2d_relu_tiled(d_p1_, d_w2_, d_b2_, d_c2_, N_, 256, H1_, W1_, 128);
        CUDA_CHECK(cudaMemcpy(d_r2_, d_c2_, nchw_size(N_,128,H1_,W1_)*sizeof(float), cudaMemcpyDeviceToDevice));
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

    // Conv3: 128->128 + ReLU3
    {
        launch_conv2d_relu_tiled(d_p2_, d_w3_, d_b3_, d_c3_, N_, 128, H2_, W2_, 128);
        CUDA_CHECK(cudaMemcpy(d_r3_, d_c3_, nchw_size(N_,128,H2_,W2_)*sizeof(float), cudaMemcpyDeviceToDevice));
    }

   

    // Upsample1: [N,128,H2,W2] -> [N,128,H1,W1]
    {
        int total = N_ * 128 * H1_ * W1_;
        dim3 grid((total + BS - 1) / BS);
        upsample2x_forward_kernel<<<grid, block>>>(
            d_r3_, d_u1_, N_, 128, H2_, W2_);
        CUDA_CHECK(cudaGetLastError());
    }

    // Conv4: 128->256 + ReLU4
    {
        launch_conv2d_relu_tiled(d_u1_, d_w4_, d_b4_, d_c4_, N_, 128, H1_, W1_, 256);
        CUDA_CHECK(cudaMemcpy(d_r4_, d_c4_, nchw_size(N_,256,H1_,W1_)*sizeof(float), cudaMemcpyDeviceToDevice));
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
    dim3 grid_data(
        (W + 15) / 16,
        (H + 15) / 16,
        N,
        Cin // Lưu ý: Loop output là Cin
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



void GPUAutoencoder::backward_pass(const float* d_dy_, float lr)
{
    const int BS = 256;
    dim3 block(BS);

    // dC5 = dY
    {
        int total = N_ * 3 * H_ * W_;
        if (d_dy_host_ptr != nullptr) {
        // [Legacy Mode] Copy từ CPU xuống (Chậm)
        CUDA_CHECK(cudaMemcpy(d_dc5_, d_dy_host_ptr, total * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            // [v1.4 Mode] Copy từ buffer GPU nội bộ (Nhanh)
            // d_dy_ là member variable chứa gradient đã tính bởi mse_loss_kernel
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

void GPUAutoencoder::backward_and_update(const Tensor& dOut, float lr)
{
    if (dOut.N() != N_ || dOut.C() != 3 || dOut.H() != H_ || dOut.W() != W_) {
        throw std::runtime_error("Gradient shape mismatch");
    }

    backward_pass(dOut.raw().data(), lr);
}

void GPUAutoencoder::save_weights(const std::string& path) const
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open " << path << " for writing\n";
        return;
    }

    auto save_tensor = [&](float* d_ptr, size_t n) {
        std::vector<float> h_buf(n);
        CUDA_CHECK(cudaMemcpy(h_buf.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
        out.write(reinterpret_cast<const char*>(h_buf.data()), n * sizeof(float));
    };

    save_tensor(d_w1_, 256 * 3 * 3 * 3);
    save_tensor(d_b1_, 256);
    save_tensor(d_w2_, 128 * 256 * 3 * 3);
    save_tensor(d_b2_, 128);
    save_tensor(d_w3_, 128 * 128 * 3 * 3);
    save_tensor(d_b3_, 128);
    save_tensor(d_w4_, 256 * 128 * 3 * 3);
    save_tensor(d_b4_, 256);
    save_tensor(d_w5_, 3 * 256 * 3 * 3);
    save_tensor(d_b5_, 3);

    out.close();
}

// ---- Kernel tính MSE Loss & Gradient dY tại chỗ ----
// Pred: d_c5_ (Output), Target: d_x_ (Input)
// dY = 2/N * (Pred - Target)


float GPUAutoencoder::train_step(const Tensor& x_host, float lr)
{
    // 1. Copy Input Host -> GPU
    set_input(x_host); 

    // 2. Forward Pass (Đã Fused & Tiled ở v1.4)
    // Dữ liệu nằm ở d_c5_ sau khi chạy xong
    forward_pass(); 

    // 3. Compute Loss & Gradient dY trên GPU
    // Reset loss accumulator
    CUDA_CHECK(cudaMemset(d_loss_accum_, 0, sizeof(float)));
    
    int total = N_ * 3 * H_ * W_;
    const int BS = 256;
    dim3 block(BS);
    dim3 grid((total + BS - 1) / BS);
    
    // Gọi kernel MSE (Input d_x_ chính là Target)
    mse_loss_kernel<<<grid, block, BS * sizeof(float)>>>(
        d_c5_, d_x_, d_dy_, d_loss_accum_, total
    );
    CUDA_CHECK(cudaGetLastError());

    // 4. Backward Pass (Đã optimized v1.3/1.4)
    // Hàm này sẽ tự lấy d_dy_ từ GPU để chạy ngược về
    backward_pass(nullptr, lr); // Tham số d_dy_ truyền nullptr vì đã có sẵn trên GPU

    // 5. Lấy giá trị Loss về CPU để báo cáo (chỉ tốn 1 lần copy 4 bytes)
    float total_loss = 0.0f;
    CUDA_CHECK(cudaMemcpy(&total_loss, d_loss_accum_, sizeof(float), cudaMemcpyDeviceToHost));
    
    return total_loss / (float)total; // Trả về Mean Squared Error
}

float GPUAutoencoder::compute_loss(const Tensor& x_host)
{
    set_input(x_host);
    forward_pass(); // Fused Forward
    
    CUDA_CHECK(cudaMemset(d_loss_accum_, 0, sizeof(float)));
    int total = N_ * 3 * H_ * W_;
    const int BS = 256;
    mse_loss_kernel<<<(total + BS - 1)/BS, BS, BS*sizeof(float)>>>(
        d_c5_, d_x_, d_dy_, d_loss_accum_, total
    );
    
    float total_loss;
    CUDA_CHECK(cudaMemcpy(&total_loss, d_loss_accum_, sizeof(float), cudaMemcpyDeviceToHost));
    return total_loss / (float)total;
}