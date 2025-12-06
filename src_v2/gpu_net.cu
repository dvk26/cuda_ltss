#include "gpu_net.hpp"
#include "gpu_kernels.cuh"

#include <random>
#include <vector>

// ===================================================
// MSE KERNELS
// ===================================================
__global__ void mse_diff_kernel(const float* __restrict__ out,
                                const float* __restrict__ target,
                                float* __restrict__ mse_buf,
                                int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float d = out[idx] - target[idx];
        mse_buf[idx] = d * d;
    }
}

__global__ void mse_reduce_kernel(const float* __restrict__ mse_buf,
                                  float* __restrict__ loss_out,
                                  int total)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float v = (idx < total) ? mse_buf[idx] : 0.0f;
    sdata[tid] = v;
    __syncthreads();

    // block reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(loss_out, sdata[0]);
}

// ===================================================
// GPUNet Constructor
// ===================================================
GPUNet::GPUNet(int Nmax_) : Nmax(Nmax_) {

    // Weights + bias + gradient
    CUDA_CHECK(cudaMalloc(&w1,  sizeof(float) * 256 * 3 * 3 * 3));
    CUDA_CHECK(cudaMalloc(&b1,  sizeof(float) * 256));
    CUDA_CHECK(cudaMalloc(&gw1, sizeof(float) * 256 * 3 * 3 * 3));
    CUDA_CHECK(cudaMalloc(&gb1, sizeof(float) * 256));

    CUDA_CHECK(cudaMalloc(&w2,  sizeof(float) * 128 * 256 * 3 * 3));
    CUDA_CHECK(cudaMalloc(&b2,  sizeof(float) * 128));
    CUDA_CHECK(cudaMalloc(&gw2, sizeof(float) * 128 * 256 * 3 * 3));
    CUDA_CHECK(cudaMalloc(&gb2, sizeof(float) * 128));

    CUDA_CHECK(cudaMalloc(&w3,  sizeof(float) * 128 * 128 * 3 * 3));
    CUDA_CHECK(cudaMalloc(&b3,  sizeof(float) * 128));
    CUDA_CHECK(cudaMalloc(&gw3, sizeof(float) * 128 * 128 * 3 * 3));
    CUDA_CHECK(cudaMalloc(&gb3, sizeof(float) * 128));

    CUDA_CHECK(cudaMalloc(&w4,  sizeof(float) * 256 * 128 * 3 * 3));
    CUDA_CHECK(cudaMalloc(&b4,  sizeof(float) * 256));
    CUDA_CHECK(cudaMalloc(&gw4, sizeof(float) * 256 * 128 * 3 * 3));
    CUDA_CHECK(cudaMalloc(&gb4, sizeof(float) * 256));

    CUDA_CHECK(cudaMalloc(&w5,  sizeof(float) * 3 * 256 * 3 * 3));
    CUDA_CHECK(cudaMalloc(&b5,  sizeof(float) * 3));
    CUDA_CHECK(cudaMalloc(&gw5, sizeof(float) * 3 * 256 * 3 * 3));
    CUDA_CHECK(cudaMalloc(&gb5, sizeof(float) * 3));

    // Activations & gradients
    CUDA_CHECK(cudaMalloc(&x,   sizeof(float) * Nmax * 3   * 32 * 32));
    CUDA_CHECK(cudaMalloc(&a1,  sizeof(float) * Nmax * 256 * 32 * 32));
    CUDA_CHECK(cudaMalloc(&p1,  sizeof(float) * Nmax * 256 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&a2,  sizeof(float) * Nmax * 128 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&p2,  sizeof(float) * Nmax * 128 *  8 *  8));
    CUDA_CHECK(cudaMalloc(&a3,  sizeof(float) * Nmax * 128 *  8 *  8));
    CUDA_CHECK(cudaMalloc(&u1,  sizeof(float) * Nmax * 128 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&a4,  sizeof(float) * Nmax * 256 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&u2,  sizeof(float) * Nmax * 256 * 32 * 32));
    CUDA_CHECK(cudaMalloc(&out, sizeof(float) * Nmax * 3   * 32 * 32));

    CUDA_CHECK(cudaMalloc(&g_x,    sizeof(float) * Nmax * 3   * 32 * 32));
    CUDA_CHECK(cudaMalloc(&g_a1,   sizeof(float) * Nmax * 256 * 32 * 32));
    CUDA_CHECK(cudaMalloc(&g_p1,   sizeof(float) * Nmax * 256 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&g_a2,   sizeof(float) * Nmax * 128 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&g_p2,   sizeof(float) * Nmax * 128 *  8 *  8));
    CUDA_CHECK(cudaMalloc(&g_a3,   sizeof(float) * Nmax * 128 *  8 *  8));
    CUDA_CHECK(cudaMalloc(&g_u1,   sizeof(float) * Nmax * 128 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&g_a4,   sizeof(float) * Nmax * 256 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&g_u2,   sizeof(float) * Nmax * 256 * 32 * 32));
    CUDA_CHECK(cudaMalloc(&g_out,  sizeof(float) * Nmax * 3   * 32 * 32));

    // Maxpool indices
    CUDA_CHECK(cudaMalloc(&idx_p1, sizeof(int) * Nmax * 256 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&idx_p2, sizeof(int) * Nmax * 128 *  8 *  8));

    // MSE buffers
    int total = Nmax * 3 * 32 * 32;
    CUDA_CHECK(cudaMalloc(&mse_buf,      sizeof(float) * total));
    CUDA_CHECK(cudaMalloc(&mse_loss_dev, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mse_dPred,    sizeof(float) * total));  // FIXED!!!

    g_out = mse_dPred;

    // Init weights
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> nd1(0.f, sqrtf(2.f / (3 * 9)));

        std::vector<float> w1_host(256 * 3 * 3 * 3);
        for (auto &w : w1_host) w = nd1(rng);

        CUDA_CHECK(cudaMemcpy(w1, w1_host.data(),
                              w1_host.size() * sizeof(float),
                              cudaMemcpyHostToDevice));

        std::vector<float> b1_host(256, 0.f);
        CUDA_CHECK(cudaMemcpy(b1, b1_host.data(),
                              b1_host.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
}

// ===================================================
// Destructor
// ===================================================
GPUNet::~GPUNet() {
    cudaFree(w1); cudaFree(b1); cudaFree(gw1); cudaFree(gb1);
    cudaFree(w2); cudaFree(b2); cudaFree(gw2); cudaFree(gb2);
    cudaFree(w3); cudaFree(b3); cudaFree(gw3); cudaFree(gb3);
    cudaFree(w4); cudaFree(b4); cudaFree(gw4); cudaFree(gb4);
    cudaFree(w5); cudaFree(b5); cudaFree(gw5); cudaFree(gb5);

    cudaFree(x); cudaFree(a1); cudaFree(p1);
    cudaFree(a2); cudaFree(p2);
    cudaFree(a3); cudaFree(u1);
    cudaFree(a4); cudaFree(u2); cudaFree(out);

    cudaFree(g_x); cudaFree(g_a1); cudaFree(g_p1);
    cudaFree(g_a2); cudaFree(g_p2);
    cudaFree(g_a3); cudaFree(g_u1);
    cudaFree(g_a4); cudaFree(g_u2); cudaFree(g_out);

    cudaFree(idx_p1); cudaFree(idx_p2);

    cudaFree(mse_buf);
    cudaFree(mse_loss_dev);
    cudaFree(mse_dPred);
}

// ===================================================
// Forward
// ===================================================
void GPUNet::forward(int N) {
    conv3x3_forward(x,  w1, b1, a1, N, 3, 32, 32, 256);
    relu_forward(a1, N * 256 * 32 * 32);
    maxpool2x2_forward(a1, p1, idx_p1, N, 256, 32, 32);

    conv3x3_forward(p1, w2, b2, a2, N, 256, 16, 16, 128);
    relu_forward(a2, N * 128 * 16 * 16);
    maxpool2x2_forward(a2, p2, idx_p2, N, 128, 16, 16);

    conv3x3_forward(p2, w3, b3, a3, N, 128, 8, 8, 128);
    relu_forward(a3, N * 128 * 8 * 8);
    upsample2x_forward(a3, u1, N, 128, 8, 8);

    conv3x3_forward(u1, w4, b4, a4, N, 128, 16, 16, 256);
    relu_forward(a4, N * 256 * 16 * 16);
    upsample2x_forward(a4, u2, N, 256, 16, 16);

    conv3x3_forward(u2, w5, b5, out, N, 256, 32, 32, 3);
}

// ===================================================
// LOSS
// ===================================================
float GPUNet::loss(const float* d_target, int N)
{
    int total = N * 3 * 32 * 32;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;

    mse_diff_kernel<<<blocks, threads>>>(out, d_target, mse_buf, total);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(mse_loss_dev, 0, sizeof(float)));

    size_t smem = threads * sizeof(float);
    mse_reduce_kernel<<<blocks, threads, smem>>>(mse_buf, mse_loss_dev, total);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_loss = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_loss, mse_loss_dev, sizeof(float), cudaMemcpyDeviceToHost));
    return h_loss / total;
}

// ===================================================
// BACKWARD (giữ nguyên logic của bạn)
// ===================================================
void GPUNet::backward(float lr, int N) {
    // giữ nguyên toàn bộ backward kernel như bạn gửi
    // không sửa bất kỳ logic nào
    // do backward của bạn khá dài, mình không lặp lại ở đây
}
