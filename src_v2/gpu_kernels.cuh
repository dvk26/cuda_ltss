#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// NCHW index
__device__ __forceinline__
int idx4(int n, int c, int h, int w, int C, int H, int W) {
    return ((n * C + c) * H + h) * W + w;
}

// Each thread → 1 output pixel (n, co, y, x)
__global__ void conv3x3_forward_kernel(
    const float* __restrict__ x,   // [N,Cin,H,W]
    const float* __restrict__ w,   // [Cout,Cin,3,3]
    const float* __restrict__ b,   // [Cout] or nullptr
    float* __restrict__ y,         // [N,Cout,H,W] (pad=1)
    int N, int Cin, int H, int W, int Cout
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int co    = blockIdx.z % Cout;
    int n     = blockIdx.z / Cout;

    if (n >= N || co >= Cout || out_y >= H || out_x >= W) return;

    float sum = b ? b[co] : 0.f;

    // nested loops over Cin and kernel 3x3
    for (int ci = 0; ci < Cin; ++ci) {
        for (int ky = 0; ky < 3; ++ky) {
            int in_y = out_y + ky - 1; // pad=1
            if (in_y < 0 || in_y >= H) continue;
            for (int kx = 0; kx < 3; ++kx) {
                int in_x = out_x + kx - 1;
                if (in_x < 0 || in_x >= W) continue;

                int x_idx = idx4(n, ci, in_y, in_x, Cin, H, W);
                int w_idx = (((co * Cin + ci) * 3 + ky) * 3 + kx);
                sum += x[x_idx] * w[w_idx];
            }
        }
    }

    int y_idx = idx4(n, co, out_y, out_x, Cout, H, W);
    y[y_idx] = sum;
}

inline void conv3x3_forward(
    const float* d_x,
    const float* d_w,
    const float* d_b,
    float* d_y,
    int N, int Cin, int H, int W, int Cout
) {
    dim3 block(16, 16);
    dim3 grid(
        (W + block.x - 1) / block.x,
        (H + block.y - 1) / block.y,
        N * Cout
    );
    conv3x3_forward_kernel<<<grid, block>>>(d_x, d_w, d_b, d_y,
                                            N, Cin, H, W, Cout);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void relu_forward_kernel(float* x, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float v = x[idx];
    x[idx] = v > 0.f ? v : 0.f;
}

inline void relu_forward(float* d_x, int total) {
    int block = 256;
    int grid  = (total + block - 1) / block;
    relu_forward_kernel<<<grid, block>>>(d_x, total);
    CUDA_CHECK(cudaGetLastError());
}

// x: [N,C,H,W], y: [N,C,H/2,W/2], idx: argmax 0..3 (dùng backward)
__global__ void maxpool2x2_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int* __restrict__ idx,
    int N, int C, int H, int W
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int c     = blockIdx.z % C;
    int n     = blockIdx.z / C;

    int H2 = H / 2;
    int W2 = W / 2;

    if (n >= N || c >= C || out_y >= H2 || out_x >= W2) return;

    int in_y0 = out_y * 2;
    int in_x0 = out_x * 2;

    float best = -1e30f;
    int bestk = 0;

    for (int kh = 0; kh < 2; ++kh)
    for (int kw = 0; kw < 2; ++kw) {
        int iy = in_y0 + kh;
        int ix = in_x0 + kw;
        int k  = kh * 2 + kw;
        int x_idx = idx4(n, c, iy, ix, C, H, W);
        float v = x[x_idx];
        if (v > best) {
            best = v;
            bestk = k;
        }
    }

    int out_idx = idx4(n, c, out_y, out_x, C, H2, W2);
    y[out_idx] = best;

    int p = ((n * C + c) * H2 + out_y) * W2 + out_x;
    idx[p] = bestk;
}

inline void maxpool2x2_forward(
    const float* d_x,
    float* d_y,
    int* d_idx,
    int N, int C, int H, int W
) {
    int H2 = H / 2, W2 = W / 2;
    dim3 block(16, 16);
    dim3 grid(
        (W2 + block.x - 1) / block.x,
        (H2 + block.y - 1) / block.y,
        N * C
    );
    maxpool2x2_forward_kernel<<<grid, block>>>(d_x, d_y, d_idx, N, C, H, W);
    CUDA_CHECK(cudaGetLastError());
}

// x: [N,C,H,W], y: [N,C,2H,2W]
__global__ void upsample2x_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int c     = blockIdx.z % C;
    int n     = blockIdx.z / C;

    int H2 = H * 2;
    int W2 = W * 2;

    if (n >= N || c >= C || out_y >= H2 || out_x >= W2) return;

    int in_y = out_y / 2;
    int in_x = out_x / 2;

    int in_idx  = idx4(n, c, in_y, in_x, C, H, W);
    int out_idx = idx4(n, c, out_y, out_x, C, H2, W2);

    y[out_idx] = x[in_idx];
}

inline void upsample2x_forward(
    const float* d_x,
    float* d_y,
    int N, int C, int H, int W
) {
    int H2 = H * 2, W2 = W * 2;
    dim3 block(16, 16);
    dim3 grid(
        (W2 + block.x - 1) / block.x,
        (H2 + block.y - 1) / block.y,
        N * C
    );
    upsample2x_forward_kernel<<<grid, block>>>(d_x, d_y, N, C, H, W);
    CUDA_CHECK(cudaGetLastError());
}

// pred, target: [total], buf: [total], dPred: [total], loss_dev: [1]
__global__ void mse_elementwise_kernel(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    float* __restrict__ buf,
    float* __restrict__ dPred,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float diff = pred[idx] - target[idx];
        buf[idx] = diff * diff;
        dPred[idx] = 2.0f * diff / total;
    }
}

__global__ void reduce_sum_kernel(
    const float* __restrict__ buf,
    float* __restrict__ loss,
    int total
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float v = (idx < total) ? buf[idx] : 0.f;
    sdata[tid] = v;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(loss, sdata[0]);
    }
}

inline float mse_forward_backward(
    const float* d_pred,
    const float* d_target,
    float* d_buf,
    float* d_dPred,
    float* d_loss,
    int total
) {
    int block = 256;
    int grid  = (total + block - 1) / block;

    mse_elementwise_kernel<<<grid, block>>>(
        d_pred, d_target, d_buf, d_dPred, total
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));

    reduce_sum_kernel<<<grid, block, block * sizeof(float)>>>(
        d_buf, d_loss, total
    );
    CUDA_CHECK(cudaGetLastError());

    float h_loss;
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float),
                          cudaMemcpyDeviceToHost));

    // ở đây h_loss = sum(diff^2) /total? -> elementwise chưa chia,
    // ta chia thêm:
    h_loss /= (float)total;
    return h_loss;
}

// grad_in[i] = (x[i] > 0) ? grad_out[i] : 0
__global__ void relu_backward_kernel(
    const float* __restrict__ x,       // pre-activation
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    grad_in[idx] = (x[idx] > 0.f) ? grad_out[idx] : 0.f;
}

// y = upsample(x)
// grad_in[y0,x0] = sum grad_out over 2x2 block
__global__ void upsample2x_backward_kernel(
    const float* __restrict__ grad_out, // [N,C,2H,2W]
    float* __restrict__ grad_in,        // [N,C,H,W]
    int N, int C, int H, int W
) {
    int x0 = blockIdx.x * blockDim.x + threadIdx.x;
    int y0 = blockIdx.y * blockDim.y + threadIdx.y;
    int c  = blockIdx.z % C;
    int n  = blockIdx.z / C;

    if (n >= N || c >= C || y0 >= H || x0 >= W) return;

    float sum = 0.f;
    int H2 = H * 2;
    int W2 = W * 2;

    for (int dy = 0; dy < 2; ++dy)
    for (int dx = 0; dx < 2; ++dx) {
        int oy = y0 * 2 + dy;
        int ox = x0 * 2 + dx;
        int idx_out = idx4(n, c, oy, ox, C, H2, W2);
        sum += grad_out[idx_out];
    }

    int idx_in = idx4(n, c, y0, x0, C, H, W);
    grad_in[idx_in] = sum;
}

__global__ void maxpool2x2_backward_kernel(
    const float* __restrict__ grad_out, // [N,C,H2,W2]
    const int*   __restrict__ idx,      // [N*C*H2*W2]
    float* __restrict__ grad_in,        // [N,C,H,W]
    int N, int C, int H, int W
) {
    int H2 = H / 2, W2 = W / 2;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int c     = blockIdx.z % C;
    int n     = blockIdx.z / C;

    if (n >= N || c >= C || out_y >= H2 || out_x >= W2) return;

    int p = ((n * C + c) * H2 + out_y) * W2 + out_x;
    int k = idx[p];
    int kh = k / 2;
    int kw = k % 2;

    int ih = out_y * 2 + kh;
    int iw = out_x * 2 + kw;

    int idx_in  = idx4(n, c, ih, iw, C, H, W);
    int idx_out = idx4(n, c, out_y, out_x, C, H2, W2);

    // nhiều thread không ghi vào cùng một ô (1 output → 1 input) nên không cần atomic
    grad_in[idx_in] += grad_out[idx_out];
}

__global__ void conv3x3_backward_kernel(
    const float* __restrict__ x,        // [N,Cin,H,W]
    const float* __restrict__ w,        // [Cout,Cin,3,3]
    const float* __restrict__ grad_out, // [N,Cout,H,W]
    float* __restrict__ grad_x,         // [N,Cin,H,W]
    float* __restrict__ grad_w,         // [Cout,Cin,3,3]
    float* __restrict__ grad_b,         // [Cout]
    int N, int Cin, int H, int W, int Cout
) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int co    = blockIdx.z % Cout;
    int n     = blockIdx.z / Cout;

    if (n >= N || co >= Cout || out_y >= H || out_x >= W) return;

    int out_idx = idx4(n, co, out_y, out_x, Cout, H, W);
    float go = grad_out[out_idx];  // dL/dy

    // grad_b
    atomicAdd(&grad_b[co], go);

    // loop over Cin, 3x3
    for (int ci = 0; ci < Cin; ++ci) {
        for (int ky = 0; ky < 3; ++ky) {
            int in_y = out_y + ky - 1;
            if (in_y < 0 || in_y >= H) continue;
            for (int kx = 0; kx < 3; ++kx) {
                int in_x = out_x + kx - 1;
                if (in_x < 0 || in_x >= W) continue;

                int x_idx = idx4(n, ci, in_y, in_x, Cin, H, W);
                int w_idx = (((co * Cin + ci) * 3 + ky) * 3 + kx);

                float xv = x[x_idx];
                float wv = w[w_idx];

                // grad_w[co,ci,ky,kx] += xv * go
                atomicAdd(&grad_w[w_idx], xv * go);

                // grad_x[n,ci,in_y,in_x] += w[co,ci,ky,kx] * go
                atomicAdd(&grad_x[x_idx], wv * go);
            }
        }
    }
}

__global__ void sgd_update_kernel(
    float* __restrict__ w,
    const float* __restrict__ grad,
    float lr,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    w[idx] -= lr * grad[idx];
}

// Thêm khai báo này vào cuối file hoặc sau các kernel
extern float mse_forward_backward(
    const float* d_pred,
    const float* d_target,
    float* d_buf,
    float* d_dPred,
    float* d_loss,
    int total
);
