#include "gpu_autoencoder.hpp"
#include <iostream>
#include <cmath>

// ====================================================
// KERNELS
// ====================================================

// ---- 1. Conv2D forward: 3x3, pad=1, stride=1, NCHW ----
// x: [N,Cin,H,W]
// w: [Cout,Cin,3,3]
// b: [Cout]
// y: [N,Cout,H,W]

#define TILE_H 16
#define TILE_W 16


__global__ void conv2d_forward_tiled_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
    int N, int Cin, int H, int W, int Cout)
{
    // blockDim = (TILE_W+2, TILE_H+2)
    const int SH_H = TILE_H + 2;
    const int SH_W = TILE_W + 2;

    int tx = threadIdx.x; // 0 .. TILE_W+1
    int ty = threadIdx.y; // 0 .. TILE_H+1

    // Góc trái trên của tile output mà block này xử lý
    int out_x0 = blockIdx.x * TILE_W;
    int out_y0 = blockIdx.y * TILE_H;

    // blockIdx.z gộp (n, c_out)
    int nc    = blockIdx.z;
    int c_out = nc % Cout;
    int n     = nc / Cout;
    if (n >= N) return;

    extern __shared__ float s[]; // kích thước = Cin * SH_H * SH_W

    // ================================
    // 1) Load tile input (kèm halo) vào shared memory
    // ================================
    int gx = out_x0 + tx - 1; // toạ độ global tương ứng (có trừ 1 để lấy halo)
    int gy = out_y0 + ty - 1;

    for (int c = 0; c < Cin; ++c) {
        float v = 0.f;
        if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
            int x_idx = ((n * Cin + c) * H + gy) * W + gx;
            v = x[x_idx];
        }
        size_t idx_s = ((size_t)c * SH_H + ty) * SH_W + tx;
        s[idx_s] = v;
    }
    __syncthreads();

    // ================================
    // 2) Chỉ các thread bên trong vùng "hữu dụng" mới tính output
    // ================================
    if (tx == 0 || tx == TILE_W + 1 || ty == 0 || ty == TILE_H + 1)
        return; // các thread ở viền chỉ dùng để load halo

    int w_out = out_x0 + (tx - 1);
    int h_out = out_y0 + (ty - 1);
    if (w_out >= W || h_out >= H) return; // block cuối có thể dư

    float sum = b[c_out];

    // kernel 3x3: kh,kw ∈ {-1,0,1}
    for (int c = 0; c < Cin; ++c) {
        for (int kh = -1; kh <= 1; ++kh) {
            for (int kw = -1; kw <= 1; ++kw) {
                int sy = ty + kh; // vị trí trong shared
                int sx = tx + kw;
                size_t idx_s = ((size_t)c * SH_H + sy) * SH_W + sx;
                float v = s[idx_s];

                int k_idx = (((c_out * Cin + c) * 3 + (kh + 1)) * 3 + (kw + 1));
                sum += v * w[k_idx];
            }
        }
    }

    int y_idx = ((n * Cout + c_out) * H + h_out) * W + w_out;
    y[y_idx] = sum;
}


__global__ void conv2d_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ y,
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

    float sum = b[c_out];

    for (int c = 0; c < Cin; ++c) {
        for (int kh = -1; kh <= 1; ++kh) {
            for (int kw = -1; kw <= 1; ++kw) {
                int ih = h_out + kh;
                int iw = w_out + kw;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;

                int x_idx = ((n * Cin + c) * H + ih) * W + iw;
                int k_idx = (((c_out * Cin + c) * 3 + (kh + 1)) * 3 + (kw + 1));
                sum += x[x_idx] * w[k_idx];
            }
        }
    }

    y[idx] = sum;
}

// ---- 2. Conv2D backward (naive, atomicAdd) ----
// dY: [N,Cout,H,W]
// x:  [N,Cin,H,W]
// w:  [Cout,Cin,3,3]
// dX: [N,Cin,H,W]
// gW: [Cout,Cin,3,3]
// gb: [Cout]
__global__ void conv2d_backward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ dY,
    float* __restrict__ dX,
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

    float grad_out = dY[idx];

    // bias grad
    atomicAdd(&gb[c_out], grad_out);

    for (int c = 0; c < Cin; ++c) {
        for (int kh = -1; kh <= 1; ++kh) {
            for (int kw = -1; kw <= 1; ++kw) {
                int ih = h_out + kh;
                int iw = w_out + kw;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;

                int x_idx = ((n * Cin + c) * H + ih) * W + iw;
                int k_idx = (((c_out * Cin + c) * 3 + (kh + 1)) * 3 + (kw + 1));

                float x_val = x[x_idx];
                float w_val = w[k_idx];

                // dW
                atomicAdd(&gW[k_idx], grad_out * x_val);
                // dX
                atomicAdd(&dX[x_idx], grad_out * w_val);
            }
        }
    }
}

// ---- 3. ReLU forward / backward (in-place-style) ----
__global__ void relu_forward_kernel(float* x, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float v = x[idx];
    x[idx] = v > 0.f ? v : 0.f;
}

__global__ void relu_backward_kernel(
    const float* __restrict__ y,   // output after ReLU
    const float* __restrict__ dY,
    float* __restrict__ dX,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    dX[idx] = (y[idx] > 0.f) ? dY[idx] : 0.f;
}

// ---- 4. MaxPool2x2 forward + backward ----
// x: [N,C,H,W], H,W even
// y: [N,C,H/2,W/2]
// mask: [N,C,H/2,W/2] (0..3, which vị trí trong 2x2 là max)
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
            int ih = h0 + kh;
            int iw = w0 + kw;
            int k = kh * 2 + kw;
            int x_idx = ((n * C + c) * H + ih) * W + iw;
            float v = x[x_idx];
            if (v > best) {
                best = v;
                bestk = k;
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
// Forward: H,W -> 2H,2W
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

// Backward: dY[2H,2W] -> dX[H,W]
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
            int h_out = h_in * 2 + kh;
            int w_out = w_in * 2 + kw;
            int y_idx = ((n * C + c) * Hout + h_out) * Wout + w_out;
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

__global__ void zero_int_kernel(int* x, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    x[idx] = 0;
}

// ---- 7. MSE loss (forward+backward) ----
// pred, target: [total_elements]
// dY: dL/dpred
// loss_accum: 1 float on device, stores SUM(diff^2)
__global__ void mse_forward_backward_kernel(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ dY,
    float* __restrict__ loss_accum,
    int total_elements,
    int batch_size) // Add batch_size argument
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.f;
    if (idx < total_elements) {
        float diff = pred[idx] - target[idx];
        
        // FIX: Scale gradient by 2/N instead of 2/Total_Elements
        // This ensures the gradient magnitude doesn't vanish as image size increases.
        dY[idx] = 2.f * diff / (float)total_elements;
        
        val = diff * diff;
    }
    sdata[tid] = val;
    __syncthreads();

    // reduction in block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(loss_accum, sdata[0]);
    }
}

// ---- 8. SGD update ----
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
    H1_ = H_ / 2; W1_ = W_ / 2;     // after Pool1
    H2_ = H1_ / 2; W2_ = W1_ / 2;   // after Pool2
    alloc_all();
    init_weights_random();
}

GPUAutoencoder::~GPUAutoencoder()
{
    free_all();
}

void GPUAutoencoder::alloc_all()
{
    // Weights sizes
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

    // Activations
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

    // Pool mask
    CUDA_CHECK(cudaMalloc(&d_mask1_, nchw_size(N_, 256, H1_, W1_) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_mask2_, nchw_size(N_, 128, H2_, W2_) * sizeof(int)));

    // Grad activations
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

    // MSE
    CUDA_CHECK(cudaMalloc(&d_target_, nchw_size(N_, 3, H_, W_) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dy_,     nchw_size(N_, 3, H_, W_) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_loss_accum_, sizeof(float)));
}

void GPUAutoencoder::free_all()
{
    auto safe_free = [](auto*& ptr) {
        if (ptr) cudaFree(ptr), ptr = nullptr;
    };

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

    safe_free(d_target_); safe_free(d_dy_); safe_free(d_loss_accum_);
}

void GPUAutoencoder::init_weights_random()
{
    std::mt19937 rng(42);

    // Helper for He Initialization: std = sqrt(2 / fan_in)
    auto init_kaiming = [&](float* d_ptr, int k_size, int cin, int cout) {
        float fan_in = (float)(cin * k_size * k_size);
        float std_dev = std::sqrt(2.0f / fan_in);
        std::normal_distribution<float> nd(0.f, std_dev);

        size_t total = (size_t)cout * cin * k_size * k_size;
        std::vector<float> h(total);
        for (size_t i = 0; i < total; ++i) h[i] = nd(rng);
        CUDA_CHECK(cudaMemcpy(d_ptr, h.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    };

    auto init_zero = [&](float* d_ptr, size_t n) {
        std::vector<float> h(n, 0.f);
        CUDA_CHECK(cudaMemcpy(d_ptr, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    };

    // W1: 3 -> 256
    init_kaiming(d_w1_, 3, 3, 256);   init_zero(d_b1_, 256);
    
    // W2: 256 -> 128
    init_kaiming(d_w2_, 3, 256, 128); init_zero(d_b2_, 128);
    
    // W3: 128 -> 128
    init_kaiming(d_w3_, 3, 128, 128); init_zero(d_b3_, 128);
    
    // W4: 128 -> 256
    init_kaiming(d_w4_, 3, 128, 256); init_zero(d_b4_, 256);
    
    // W5: 256 -> 3
    init_kaiming(d_w5_, 3, 256, 3);   init_zero(d_b5_, 3);
}

// --------- copy_weights_from_cpu / to_cpu ----------
// Lưu ý: Autoencoder CPU của bạn có hàm weights() / biases() gì đó.
// Ở đây mình giả sử Conv2D có:
//   const std::vector<float>& weights() const;
//   const std::vector<float>& biases() const;
// Bạn chỉnh lại cho khớp nếu prototype khác.

void GPUAutoencoder::copy_weights_from_cpu(const Autoencoder& cpu)
{
    // ===== Conv1: 3 -> 256 =====
    {
        const auto& w = cpu.c1().weights();  // [256,3,3,3] flattened
        const auto& b = cpu.c1().bias();     // [256]
        CUDA_CHECK(cudaMemcpy(d_w1_, w.data(),
                              w.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b1_, b.data(),
                              b.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // ===== Conv2: 256 -> 128 =====
    {
        const auto& w = cpu.c2().weights();  // [128,256,3,3]
        const auto& b = cpu.c2().bias();     // [128]
        CUDA_CHECK(cudaMemcpy(d_w2_, w.data(),
                              w.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b2_, b.data(),
                              b.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // ===== Conv3: 128 -> 128 =====
    {
        const auto& w = cpu.c3().weights();  // [128,128,3,3]
        const auto& b = cpu.c3().bias();     // [128]
        CUDA_CHECK(cudaMemcpy(d_w3_, w.data(),
                              w.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b3_, b.data(),
                              b.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // ===== Conv4: 128 -> 256 =====
    {
        const auto& w = cpu.c4().weights();  // [256,128,3,3]
        const auto& b = cpu.c4().bias();     // [256]
        CUDA_CHECK(cudaMemcpy(d_w4_, w.data(),
                              w.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b4_, b.data(),
                              b.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // ===== Conv5: 256 -> 3 =====
    {
        const auto& w = cpu.c5().weights();  // [3,256,3,3]
        const auto& b = cpu.c5().bias();     // [3]
        CUDA_CHECK(cudaMemcpy(d_w5_, w.data(),
                              w.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b5_, b.data(),
                              b.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
}

void GPUAutoencoder::copy_weights_to_cpu(Autoencoder& cpu) const
{
    // ===== Conv1 =====
    {
        auto& w = cpu.c1().weights();  // non-const
        auto& b = cpu.c1().bias();
        CUDA_CHECK(cudaMemcpy(w.data(), d_w1_,
                              w.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), d_b1_,
                              b.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    // ===== Conv2 =====
    {
        auto& w = cpu.c2().weights();
        auto& b = cpu.c2().bias();
        CUDA_CHECK(cudaMemcpy(w.data(), d_w2_,
                              w.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), d_b2_,
                              b.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    // ===== Conv3 =====
    {
        auto& w = cpu.c3().weights();
        auto& b = cpu.c3().bias();
        CUDA_CHECK(cudaMemcpy(w.data(), d_w3_,
                              w.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), d_b3_,
                              b.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    // ===== Conv4 =====
    {
        auto& w = cpu.c4().weights();
        auto& b = cpu.c4().bias();
        CUDA_CHECK(cudaMemcpy(w.data(), d_w4_,
                              w.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), d_b4_,
                              b.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    // ===== Conv5 =====
    {
        auto& w = cpu.c5().weights();
        auto& b = cpu.c5().bias();
        CUDA_CHECK(cudaMemcpy(w.data(), d_w5_,
                              w.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), d_b5_,
                              b.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
}

// --------- set_input / set_target ----------

void GPUAutoencoder::set_input(const Tensor& x_host)
{
    size_t sz = nchw_size(N_, 3, H_, W_);
    CUDA_CHECK(cudaMemcpy(d_x_, x_host.raw().data(), sz * sizeof(float), cudaMemcpyHostToDevice));
}

void GPUAutoencoder::set_target(const Tensor& target_host)
{
    size_t sz = nchw_size(N_, 3, H_, W_);
    CUDA_CHECK(cudaMemcpy(d_target_, target_host.raw().data(), sz * sizeof(float), cudaMemcpyHostToDevice));
}

// --------- forward_pass: d_x_ -> d_c5_ + MSE (d_target_) ----------

void GPUAutoencoder::forward_pass()
{
    const int BS = 256;
    dim3 block(BS);

    // ----- Conv1 -----
    {
        dim3 block(TILE_W + 2, TILE_H + 2); // +2 để có halo
        dim3 grid((W_ + TILE_W - 1) / TILE_W,
                  (H_ + TILE_H - 1) / TILE_H,
                  N_ * 256); // gộp (n, c_out) trong blockIdx.z

        // Cin = 3 cho Conv1
        size_t shm = (size_t)3 * (TILE_H + 2) * (TILE_W + 2) * sizeof(float);

        conv2d_forward_tiled_kernel<<<grid, block, shm>>>(
            d_x_, d_w1_, d_b1_, d_c1_,
            N_, 3, H_, W_, 256);

    }

    // ReLU1 (in-place trên d_c1_, lưu ra d_r1_)
    {
        // copy d_c1_ -> d_r1_
        CUDA_CHECK(cudaMemcpy(d_r1_, d_c1_, nchw_size(N_,256,H_,W_)*sizeof(float),
                              cudaMemcpyDeviceToDevice));
        int total = (int)nchw_size(N_, 256, H_, W_);
        dim3 grid((total + BS - 1) / BS);
        relu_forward_kernel<<<grid, block>>>(d_r1_, total);
    }

    // Pool1: d_r1_ -> d_p1_
    {
        int total = (int)nchw_size(N_, 256, H1_, W1_);
        dim3 grid((total + BS - 1) / BS);
        maxpool2x2_forward_kernel<<<grid, block>>>(
            d_r1_, d_p1_, d_mask1_,
            N_, 256, H_, W_);
    }

    // Conv2: 256->128
    {
        int total = (int)nchw_size(N_, 128, H1_, W1_);
        dim3 grid((total + BS - 1) / BS);
        conv2d_forward_kernel<<<grid, block>>>(
            d_p1_, d_w2_, d_b2_, d_c2_,
            N_, 256, H1_, W1_, 128);
    }

    // ReLU2
    {
        CUDA_CHECK(cudaMemcpy(d_r2_, d_c2_, nchw_size(N_,128,H1_,W1_)*sizeof(float),
                              cudaMemcpyDeviceToDevice));
        int total = (int)nchw_size(N_, 128, H1_, W1_);
        dim3 grid((total + BS - 1) / BS);
        relu_forward_kernel<<<grid, block>>>(d_r2_, total);
    }

    // Pool2: d_r2_ -> d_p2_ (latent)
    {
        int total = (int)nchw_size(N_, 128, H2_, W2_);
        dim3 grid((total + BS - 1) / BS);
        maxpool2x2_forward_kernel<<<grid, block>>>(
            d_r2_, d_p2_, d_mask2_,
            N_, 128, H1_, W1_);
    }

    // ---- Decoder ----

    // Conv3: 128->128
    {
        int total = (int)nchw_size(N_, 128, H2_, W2_);
        dim3 grid((total + BS - 1) / BS);
        conv2d_forward_kernel<<<grid, block>>>(
            d_p2_, d_w3_, d_b3_, d_c3_,
            N_, 128, H2_, W2_, 128);
    }

    // ReLU3
    {
        CUDA_CHECK(cudaMemcpy(d_r3_, d_c3_, nchw_size(N_,128,H2_,W2_)*sizeof(float),
                              cudaMemcpyDeviceToDevice));
        int total = (int)nchw_size(N_, 128, H2_, W2_);
        dim3 grid((total + BS - 1) / BS);
        relu_forward_kernel<<<grid, block>>>(d_r3_, total);
    }

    // Upsample1: [N,128,H2,W2] -> [N,128,H1,W1]
    {
        int total = (int)nchw_size(N_, 128, H1_, W1_);
        dim3 grid((total + BS - 1) / BS);
        upsample2x_forward_kernel<<<grid, block>>>(
            d_r3_, d_u1_, N_, 128, H2_, W2_);
    }

    // Conv4: 128->256
    {
        int total = (int)nchw_size(N_, 256, H1_, W1_);
        dim3 grid((total + BS - 1) / BS);
        conv2d_forward_kernel<<<grid, block>>>(
            d_u1_, d_w4_, d_b4_, d_c4_,
            N_, 128, H1_, W1_, 256);
    }

    // ReLU4
    {
        CUDA_CHECK(cudaMemcpy(d_r4_, d_c4_, nchw_size(N_,256,H1_,W1_)*sizeof(float),
                              cudaMemcpyDeviceToDevice));
        int total = (int)nchw_size(N_, 256, H1_, W1_);
        dim3 grid((total + BS - 1) / BS);
        relu_forward_kernel<<<grid, block>>>(d_r4_, total);
    }

    // Upsample2: [N,256,H1,W1] -> [N,256,H,W]
    {
        int total = (int)nchw_size(N_, 256, H_, W_);
        dim3 grid((total + BS - 1) / BS);
        upsample2x_forward_kernel<<<grid, block>>>(
            d_r4_, d_u2_, N_, 256, H1_, W1_);
    }

    // Conv5: 256->3 (output)
    {
        int total = (int)nchw_size(N_, 3, H_, W_);
        dim3 grid((total + BS - 1) / BS);
        conv2d_forward_kernel<<<grid, block>>>(
            d_u2_, d_w5_, d_b5_, d_c5_,
            N_, 256, H_, W_, 3);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

// --------- forward_internal (legacy): d_x_ -> d_c5_ + MSE (d_target_) + return loss --------

float GPUAutoencoder::forward_internal()
{
    // First run the forward pass
    forward_pass();

    // Then compute MSE + d_dy_ (dL/doutput)
    // d_target_ đã được set trước đó
    const int BS = 256;
    dim3 block(BS);
    
    CUDA_CHECK(cudaMemset(d_loss_accum_, 0, sizeof(float)));
    {
        int total = (int)nchw_size(N_, 3, H_, W_);
        dim3 grid((total + BS - 1) / BS);
        size_t shm = BS * sizeof(float);
        mse_forward_backward_kernel<<<grid, block, shm>>>(
            d_c5_, d_target_, d_dy_, d_loss_accum_, total, N_);
    }

    float loss_sum;
    CUDA_CHECK(cudaMemcpy(&loss_sum, d_loss_accum_, sizeof(float), cudaMemcpyDeviceToHost));

    int total_elem = (int)nchw_size(N_, 3, H_, W_);
    float loss = loss_sum / (float)total_elem;
    return loss;
}

// --------- backward_pass: backprop + SGD update (new cleaner API) --------

void GPUAutoencoder::backward_pass(float lr)
{
    const int BS = 256;
    dim3 block(BS);

    // dC5 = dY
    {
        int total = (int)nchw_size(N_, 3, H_, W_);
        CUDA_CHECK(cudaMemcpy(d_dc5_, d_dy_,
                              total * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }

    // Zero ALL gradient buffers
    auto zero_buf = [&](float* p, size_t n) {
        int total = (int)n;
        dim3 grid((total + BS - 1) / BS);
        zero_kernel<<<grid, block>>>(p, total);
    };

    zero_buf(d_gw1_, 256 * 3 * 3 * 3);
    zero_buf(d_gb1_, 256);
    zero_buf(d_gw2_, 128 * 256 * 3 * 3);
    zero_buf(d_gb2_, 128);
    zero_buf(d_gw3_, 128 * 128 * 3 * 3);
    zero_buf(d_gb3_, 128);
    zero_buf(d_gw4_, 256 * 128 * 3 * 3);
    zero_buf(d_gb4_, 256);
    zero_buf(d_gw5_, 3   * 256 * 3 * 3);
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

    // ----- Backward qua Conv5 -----
    {
        int total = (int)nchw_size(N_, 3, H_, W_);
        dim3 grid((total + BS - 1) / BS);
        conv2d_backward_kernel<<<grid, block>>>(
            d_u2_, d_w5_, d_dc5_, d_du2_, d_gw5_, d_gb5_,
            N_, 256, H_, W_, 3);
    }

    // Up2 backward: du2_ -> dr4_
    {
        int total = (int)nchw_size(N_, 256, H1_, W1_);
        dim3 grid((total + BS - 1) / BS);
        upsample2x_backward_kernel<<<grid, block>>>(
            d_du2_, d_dr4_, N_, 256, H1_, W1_);
    }

    // ReLU4 backward: dr4_ + output r4_ -> dc4_
    {
        int total = (int)nchw_size(N_,256,H1_,W1_);
        dim3 grid((total + BS - 1) / BS);
        relu_backward_kernel<<<grid, block>>>(d_r4_, d_dr4_, d_dc4_, total);
    }

    // Conv4 backward: dc4_ -> du1_ + gw4_, gb4_
    {
        int total = (int)nchw_size(N_,256,H1_,W1_);
        dim3 grid((total + BS - 1) / BS);
        conv2d_backward_kernel<<<grid, block>>>(
            d_u1_, d_w4_, d_dc4_, d_du1_, d_gw4_, d_gb4_,
            N_, 128, H1_, W1_, 256);
    }

    // Up1 backward: du1_ -> dr3_
    {
        int total = (int)nchw_size(N_,128,H2_,W2_);
        dim3 grid((total + BS - 1) / BS);
        upsample2x_backward_kernel<<<grid, block>>>(
            d_du1_, d_dr3_, N_, 128, H2_, W2_);
    }

    // ReLU3 backward: dr3_ -> dc3_
    {
        int total = (int)nchw_size(N_,128,H2_,W2_);
        dim3 grid((total + BS - 1) / BS);
        relu_backward_kernel<<<grid, block>>>(d_r3_, d_dr3_, d_dc3_, total);
    }

    // Conv3 backward: dc3_ -> dp2_ + gw3_, gb3_
    {
        int total = (int)nchw_size(N_,128,H2_,W2_);
        dim3 grid((total + BS - 1) / BS);
        conv2d_backward_kernel<<<grid, block>>>(
            d_p2_, d_w3_, d_dc3_, d_dp2_, d_gw3_, d_gb3_,
            N_, 128, H2_, W2_, 128);
    }

    // Pool2 backward: dp2_ + mask2_ -> dr2_
    {
        int total = (int)nchw_size(N_,128,H1_,W1_);
        dim3 grid((total + BS - 1) / BS);
        maxpool2x2_backward_kernel<<<grid, block>>>(
            d_dp2_, d_mask2_, d_dr2_,
            N_, 128, H1_, W1_);
    }

    // ReLU2 backward: dr2_ -> dc2_
    {
        int total = (int)nchw_size(N_,128,H1_,W1_);
        dim3 grid((total + BS - 1) / BS);
        relu_backward_kernel<<<grid, block>>>(d_r2_, d_dr2_, d_dc2_, total);
    }

    // Conv2 backward: dc2_ -> dp1_ + gw2_, gb2_
    {
        int total = (int)nchw_size(N_,128,H1_,W1_);
        dim3 grid((total + BS - 1) / BS);
        conv2d_backward_kernel<<<grid, block>>>(
            d_p1_, d_w2_, d_dc2_, d_dp1_, d_gw2_, d_gb2_,
            N_, 256, H1_, W1_, 128);
    }

    // Pool1 backward: dp1_ + mask1_ -> dr1_
    {
        int total = (int)nchw_size(N_,256,H_,W_);
        dim3 grid((total + BS - 1) / BS);
        maxpool2x2_backward_kernel<<<grid, block>>>(
            d_dp1_, d_mask1_, d_dr1_,
            N_, 256, H_, W_);
    }

    // ReLU1 backward: dr1_ -> dc1_
    {
        int total = (int)nchw_size(N_,256,H_,W_);
        dim3 grid((total + BS - 1) / BS);
        relu_backward_kernel<<<grid, block>>>(d_r1_, d_dr1_, d_dc1_, total);
    }

    // Conv1 backward: dc1_ -> dx_ + gw1_, gb1_
    {
        int total = (int)nchw_size(N_,256,H_,W_);
        dim3 grid((total + BS - 1) / BS);
        conv2d_backward_kernel<<<grid, block>>>(
            d_x_, d_w1_, d_dc1_, d_dx_, d_gw1_, d_gb1_,
            N_, 3, H_, W_, 256);
    }

    // ----- SGD update -----
    auto sgd = [&](float* w, float* gw, int n) {
        int total = n;
        dim3 grid((total + BS - 1) / BS);
        sgd_update_kernel<<<grid, block>>>(w, gw, lr, total);
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

// --------- backward_internal (legacy): wrapper for backward_pass --------

float GPUAutoencoder::backward_internal(float lr)
{
    backward_pass(lr);
    return 0.f; // không cần trả gì, loss đã tính ở forward_internal
}

// --------- New public API: encode / decode / forward / backward_and_update ----------

Tensor GPUAutoencoder::encode(const Tensor& x_host)
{
    if (x_host.N() != N_ || x_host.C() != 3 || x_host.H() != H_ || x_host.W() != W_) {
        throw std::runtime_error("GPUAutoencoder::encode: input shape mismatch");
    }

    set_input(x_host);

    // Forward encoder only: x -> c1 -> r1 -> p1 -> c2 -> r2 -> p2 (latent)
    const int BS = 256;
    dim3 block(BS);

    // Conv1
    {
        int total = (int)nchw_size(N_, 256, H_, W_);
        dim3 grid((total + BS - 1) / BS);
        conv2d_forward_kernel<<<grid, block>>>(
            d_x_, d_w1_, d_b1_, d_c1_,
            N_, 3, H_, W_, 256);
    }

    // ReLU1
    {
        size_t sz = nchw_size(N_, 256, H_, W_) * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_r1_, d_c1_, sz, cudaMemcpyDeviceToDevice));
        int total = (int)nchw_size(N_, 256, H_, W_);
        dim3 grid((total + BS - 1) / BS);
        relu_forward_kernel<<<grid, block>>>(d_r1_, total);
    }

    // Pool1
    {
        int total = (int)nchw_size(N_, 256, H1_, W1_);
        dim3 grid((total + BS - 1) / BS);
        maxpool2x2_forward_kernel<<<grid, block>>>(
            d_r1_, d_p1_, d_mask1_,
            N_, 256, H_, W_);
    }

    // Conv2
    {
        int total = (int)nchw_size(N_, 128, H1_, W1_);
        dim3 grid((total + BS - 1) / BS);
        conv2d_forward_kernel<<<grid, block>>>(
            d_p1_, d_w2_, d_b2_, d_c2_,
            N_, 256, H1_, W1_, 128);
    }

    // ReLU2
    {
        size_t sz = nchw_size(N_, 128, H1_, W1_) * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_r2_, d_c2_, sz, cudaMemcpyDeviceToDevice));
        int total = (int)nchw_size(N_, 128, H1_, W1_);
        dim3 grid((total + BS - 1) / BS);
        relu_forward_kernel<<<grid, block>>>(d_r2_, total);
    }

    // Pool2 (latent)
    {
        int total = (int)nchw_size(N_, 128, H2_, W2_);
        dim3 grid((total + BS - 1) / BS);
        maxpool2x2_forward_kernel<<<grid, block>>>(
            d_r2_, d_p2_, d_mask2_,
            N_, 128, H1_, W1_);
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
        throw std::runtime_error("GPUAutoencoder::decode: latent shape mismatch");
    }

    // Copy latent to GPU
    size_t sz = nchw_size(N_, 128, H2_, W2_) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_p2_, z_host.raw().data(), sz, cudaMemcpyHostToDevice));

    // Forward decoder only: z -> c3 -> r3 -> u1 -> c4 -> r4 -> u2 -> c5
    const int BS = 256;
    dim3 block(BS);

    // Conv3
    {
        int total = (int)nchw_size(N_, 128, H2_, W2_);
        dim3 grid((total + BS - 1) / BS);
        conv2d_forward_kernel<<<grid, block>>>(
            d_p2_, d_w3_, d_b3_, d_c3_,
            N_, 128, H2_, W2_, 128);
    }

    // ReLU3
    {
        size_t sz_tmp = nchw_size(N_, 128, H2_, W2_) * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_r3_, d_c3_, sz_tmp, cudaMemcpyDeviceToDevice));
        int total = (int)nchw_size(N_, 128, H2_, W2_);
        dim3 grid((total + BS - 1) / BS);
        relu_forward_kernel<<<grid, block>>>(d_r3_, total);
    }

    // Upsample1
    {
        int total = (int)nchw_size(N_, 128, H1_, W1_);
        dim3 grid((total + BS - 1) / BS);
        upsample2x_forward_kernel<<<grid, block>>>(
            d_r3_, d_u1_,
            N_, 128, H2_, W2_);
    }

    // Conv4
    {
        int total = (int)nchw_size(N_, 256, H1_, W1_);
        dim3 grid((total + BS - 1) / BS);
        conv2d_forward_kernel<<<grid, block>>>(
            d_u1_, d_w4_, d_b4_, d_c4_,
            N_, 128, H1_, W1_, 256);
    }

    // ReLU4
    {
        size_t sz_tmp = nchw_size(N_, 256, H1_, W1_) * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_r4_, d_c4_, sz_tmp, cudaMemcpyDeviceToDevice));
        int total = (int)nchw_size(N_, 256, H1_, W1_);
        dim3 grid((total + BS - 1) / BS);
        relu_forward_kernel<<<grid, block>>>(d_r4_, total);
    }

    // Upsample2
    {
        int total = (int)nchw_size(N_, 256, H_, W_);
        dim3 grid((total + BS - 1) / BS);
        upsample2x_forward_kernel<<<grid, block>>>(
            d_r4_, d_u2_,
            N_, 256, H1_, W1_);
    }

    // Conv5
    {
        int total = (int)nchw_size(N_, 3, H_, W_);
        dim3 grid((total + BS - 1) / BS);
        conv2d_forward_kernel<<<grid, block>>>(
            d_u2_, d_w5_, d_b5_, d_c5_,
            N_, 256, H_, W_, 3);
    }

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
        throw std::runtime_error("GPUAutoencoder::forward: input shape mismatch");
    }

    set_input(x_host);
    
    // Set target = input for loss computation (if needed)
    size_t sz = nchw_size(N_, 3, H_, W_) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_target_, d_x_, sz, cudaMemcpyDeviceToDevice));
    
    forward_pass();

    // Copy output to host
    Tensor output(N_, 3, H_, W_);
    CUDA_CHECK(cudaMemcpy(output.raw().data(), d_c5_, sz, cudaMemcpyDeviceToHost));

    return output;
}

void GPUAutoencoder::backward_and_update(const Tensor& dOut, float lr)
{
    if (dOut.N() != N_ || dOut.C() != 3 || dOut.H() != H_ || dOut.W() != W_) {
        throw std::runtime_error("GPUAutoencoder::backward_and_update: gradient shape mismatch");
    }

    // Copy gradient to GPU
    size_t sz = nchw_size(N_, 3, H_, W_) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(d_dy_, dOut.raw().data(), sz, cudaMemcpyHostToDevice));

    // Run backward pass with SGD updates
    backward_pass(lr);
}
