#include "gpu_net.hpp"
#include "gpu_kernels.cuh"
#include <random>
#include <vector>

GPUNet::GPUNet(int Nmax_) : Nmax(Nmax_) {
    // 1) cudaMalloc weights/bias + gradients
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

    // 2) cudaMalloc activations & gradients theo Nmax
    CUDA_CHECK(cudaMalloc(&x,   sizeof(float) * Nmax * 3   * 32 * 32));
    CUDA_CHECK(cudaMalloc(&a1,  sizeof(float) * Nmax * 256 * 32 * 32));
    CUDA_CHECK(cudaMalloc(&a1_relu, sizeof(float) * Nmax * 256 * 32 * 32));
    CUDA_CHECK(cudaMalloc(&p1,  sizeof(float) * Nmax * 256 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&a2,  sizeof(float) * Nmax * 128 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&a2_relu, sizeof(float) * Nmax * 128 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&p2,  sizeof(float) * Nmax * 128 * 8 * 8));
    CUDA_CHECK(cudaMalloc(&a3,  sizeof(float) * Nmax * 128 * 8 * 8));
    CUDA_CHECK(cudaMalloc(&a3_relu, sizeof(float) * Nmax * 128 * 8 * 8));
    CUDA_CHECK(cudaMalloc(&u1,  sizeof(float) * Nmax * 128 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&a4,  sizeof(float) * Nmax * 256 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&a4_relu, sizeof(float) * Nmax * 256 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&u2,  sizeof(float) * Nmax * 256 * 32 * 32));
    CUDA_CHECK(cudaMalloc(&out, sizeof(float) * Nmax * 3 * 32 * 32));

    CUDA_CHECK(cudaMalloc(&g_x,   sizeof(float) * Nmax * 3   * 32 * 32));
    CUDA_CHECK(cudaMalloc(&g_a1,  sizeof(float) * Nmax * 256 * 32 * 32));
    CUDA_CHECK(cudaMalloc(&g_a1_relu, sizeof(float) * Nmax * 256 * 32 * 32));
    CUDA_CHECK(cudaMalloc(&g_p1,  sizeof(float) * Nmax * 256 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&g_a2,  sizeof(float) * Nmax * 128 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&g_a2_relu, sizeof(float) * Nmax * 128 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&g_p2,  sizeof(float) * Nmax * 128 * 8 * 8));
    CUDA_CHECK(cudaMalloc(&g_a3,  sizeof(float) * Nmax * 128 * 8 * 8));
    CUDA_CHECK(cudaMalloc(&g_a3_relu, sizeof(float) * Nmax * 128 * 8 * 8));
    CUDA_CHECK(cudaMalloc(&g_u1,  sizeof(float) * Nmax * 128 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&g_a4,  sizeof(float) * Nmax * 256 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&g_a4_relu, sizeof(float) * Nmax * 256 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&g_u2,  sizeof(float) * Nmax * 256 * 32 * 32));
    CUDA_CHECK(cudaMalloc(&g_out, sizeof(float) * Nmax * 3 * 32 * 32));

    // 3) idx_p1, idx_p2
    CUDA_CHECK(cudaMalloc(&idx_p1, sizeof(int) * Nmax * 256 * 16 * 16));
    CUDA_CHECK(cudaMalloc(&idx_p2, sizeof(int) * Nmax * 128 *  8 *  8));

    // 4) MSE buffers
    int total = Nmax * 3 * 32 * 32;
    CUDA_CHECK(cudaMalloc(&mse_buf,     sizeof(float) * total));
    CUDA_CHECK(cudaMalloc(&mse_dPred,   sizeof(float) * total));
    CUDA_CHECK(cudaMalloc(&mse_loss_dev, sizeof(float)));

    g_out = mse_dPred; // dL/dOut

    // 5) init weights trên host rồi copy lên (dùng std::mt19937 giống Conv2D CPU)
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> nd1(0.f, std::sqrt(2.f/(3*9)));
        std::vector<float> w1_host(256*3*3*3);
        for (auto& w : w1_host) w = nd1(rng);
        CUDA_CHECK(cudaMemcpy(w1, w1_host.data(), w1_host.size()*sizeof(float), cudaMemcpyHostToDevice));
        std::vector<float> b1_host(256, 0.f);
        CUDA_CHECK(cudaMemcpy(b1, b1_host.data(), b1_host.size()*sizeof(float), cudaMemcpyHostToDevice));
        // Tương tự cho w2..w5, b2..b5
        // ...
    }
}

GPUNet::~GPUNet() {
    cudaFree(w1); cudaFree(b1); cudaFree(gw1); cudaFree(gb1);
    cudaFree(w2); cudaFree(b2); cudaFree(gw2); cudaFree(gb2);
    cudaFree(w3); cudaFree(b3); cudaFree(gw3); cudaFree(gb3);
    cudaFree(w4); cudaFree(b4); cudaFree(gw4); cudaFree(gb4);
    cudaFree(w5); cudaFree(b5); cudaFree(gw5); cudaFree(gb5);

    cudaFree(x); cudaFree(a1); cudaFree(a1_relu); cudaFree(p1);
    cudaFree(a2); cudaFree(a2_relu); cudaFree(p2);
    cudaFree(a3); cudaFree(a3_relu); cudaFree(u1);
    cudaFree(a4); cudaFree(a4_relu); cudaFree(u2); cudaFree(out);

    cudaFree(g_x); cudaFree(g_a1); cudaFree(g_a1_relu); cudaFree(g_p1);
    cudaFree(g_a2); cudaFree(g_a2_relu); cudaFree(g_p2);
    cudaFree(g_a3); cudaFree(g_a3_relu); cudaFree(g_u1);
    cudaFree(g_a4); cudaFree(g_a4_relu); cudaFree(g_u2); cudaFree(g_out);

    cudaFree(idx_p1); cudaFree(idx_p2);
    cudaFree(mse_buf); cudaFree(mse_dPred); cudaFree(mse_loss_dev);
}

void GPUNet::forward(int N) {
    // 1) conv1: x -> a1
    conv3x3_forward(x, w1, b1, a1, N, 3, 32, 32, 256);
    relu_forward(a1, N * 256 * 32 * 32);
    maxpool2x2_forward(a1, p1, idx_p1, N, 256, 32, 32);

    // 2) conv2: p1 -> a2
    conv3x3_forward(p1, w2, b2, a2, N, 256, 16, 16, 128);
    relu_forward(a2, N * 128 * 16 * 16);
    maxpool2x2_forward(a2, p2, idx_p2, N, 128, 16, 16);

    // 3) conv3: p2 -> a3
    conv3x3_forward(p2, w3, b3, a3, N, 128, 8, 8, 128);
    relu_forward(a3, N * 128 * 8 * 8);
    upsample2x_forward(a3, u1, N, 128, 8, 8);

    // 4) conv4: u1 -> a4
    conv3x3_forward(u1, w4, b4, a4, N, 128, 16, 16, 256);
    relu_forward(a4, N * 256 * 16 * 16);
    upsample2x_forward(a4, u2, N, 256, 16, 16);

    // 5) conv5: u2 -> out
    conv3x3_forward(u2, w5, b5, out, N, 256, 32, 32, 3);
}

float GPUNet::loss(const float* d_target, int N) {
    int total = N * 3 * 32 * 32;
    return mse_forward_backward(
        out, d_target,
        mse_buf, mse_dPred,
        mse_loss_dev, total
    );
}

void GPUNet::backward(float lr, int N) {
    int total_out = N * 3 * 32 * 32;
    int total_u2  = N * 256 * 32 * 32;
    int total_a4  = N * 256 * 16 * 16;
    int total_u1  = N * 128 * 16 * 16;
    int total_a3  = N * 128 * 8 * 8;
    int total_p2  = N * 128 * 8 * 8;
    int total_a2  = N * 128 * 16 * 16;
    int total_p1  = N * 256 * 16 * 16;
    int total_a1  = N * 256 * 32 * 32;

    // Conv5 backward: out, u2
    CUDA_CHECK(cudaMemset(g_u2, 0, total_u2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(gw5, 0, 3*256*3*3*sizeof(float)));
    CUDA_CHECK(cudaMemset(gb5, 0, 3*sizeof(float)));
    conv3x3_backward_kernel<<<dim3(16,16,N*3),256>>>(
        u2, w5, mse_dPred, g_u2, gw5, gb5, N, 256, 32, 32, 3);

    // Up2 backward: g_u2 -> g_a4
    CUDA_CHECK(cudaMemset(g_a4, 0, total_a4 * sizeof(float)));
    upsample2x_backward_kernel<<<dim3(16,16,N*256),256>>>(
        g_u2, g_a4, N, 256, 16, 16);

    // ReLU4 backward: g_a4 <- a4, grad_out=g_a4
    relu_backward_kernel<<<(total_a4+255)/256,256>>>(
        a4, g_a4, g_a4, total_a4);

    // Conv4 backward: a4, u1
    CUDA_CHECK(cudaMemset(g_u1, 0, total_u1 * sizeof(float)));
    CUDA_CHECK(cudaMemset(gw4, 0, 256*128*3*3*sizeof(float)));
    CUDA_CHECK(cudaMemset(gb4, 0, 256*sizeof(float)));
    conv3x3_backward_kernel<<<dim3(16,16,N*256),256>>>(
        u1, w4, g_a4, g_u1, gw4, gb4, N, 128, 16, 16, 256);

    // Up1 backward: g_u1 -> g_a3
    CUDA_CHECK(cudaMemset(g_a3, 0, total_a3 * sizeof(float)));
    upsample2x_backward_kernel<<<dim3(8,8,N*128),256>>>(
        g_u1, g_a3, N, 128, 8, 8);

    // ReLU3 backward: g_a3 <- a3, grad_out=g_a3
    relu_backward_kernel<<<(total_a3+255)/256,256>>>(
        a3, g_a3, g_a3, total_a3);

    // Conv3 backward: a3, p2
    CUDA_CHECK(cudaMemset(g_p2, 0, total_p2 * sizeof(float)));
    CUDA_CHECK(cudaMemset(gw3, 0, 128*128*3*3*sizeof(float)));
    CUDA_CHECK(cudaMemset(gb3, 0, 128*sizeof(float)));
    conv3x3_backward_kernel<<<dim3(8,8,N*128),256>>>(
        p2, w3, g_a3, g_p2, gw3, gb3, N, 128, 8, 8, 128);

    // MaxPool2 backward: g_p2 -> g_a2
    CUDA_CHECK(cudaMemset(g_a2, 0, total_a2 * sizeof(float)));
    maxpool2x2_backward_kernel<<<dim3(16,16,N*128),256>>>(
        g_p2, idx_p2, g_a2, N, 128, 16, 16);

    // ReLU2 backward: g_a2 <- a2, grad_out=g_a2
    relu_backward_kernel<<<(total_a2+255)/256,256>>>(
        a2, g_a2, g_a2, total_a2);

    // Conv2 backward: a2, p1
    CUDA_CHECK(cudaMemset(g_p1, 0, total_p1 * sizeof(float)));
    CUDA_CHECK(cudaMemset(gw2, 0, 128*256*3*3*sizeof(float)));
    CUDA_CHECK(cudaMemset(gb2, 0, 128*sizeof(float)));
    conv3x3_backward_kernel<<<dim3(16,16,N*128),256>>>(
        p1, w2, g_a2, g_p1, gw2, gb2, N, 256, 16, 16, 128);

    // MaxPool1 backward: g_p1 -> g_a1
    CUDA_CHECK(cudaMemset(g_a1, 0, total_a1 * sizeof(float)));
    maxpool2x2_backward_kernel<<<dim3(16,16,N*256),256>>>(
        g_p1, idx_p1, g_a1, N, 256, 32, 32);

    // ReLU1 backward: g_a1 <- a1, grad_out=g_a1
    relu_backward_kernel<<<(total_a1+255)/256,256>>>(
        a1, g_a1, g_a1, total_a1);

    // Conv1 backward: x, a1
    CUDA_CHECK(cudaMemset(gw1, 0, 256*3*3*3*sizeof(float)));
    CUDA_CHECK(cudaMemset(gb1, 0, 256*sizeof(float)));
    conv3x3_backward_kernel<<<dim3(16,16,N*256),256>>>(
        x, w1, g_a1, nullptr, gw1, gb1, N, 3, 32, 32, 256);

    // ---- SGD update ----
    sgd_update_kernel<<<(256*3*3*3+255)/256,256>>>(w1, gw1, lr, 256*3*3*3);
    sgd_update_kernel<<<(256+255)/256,256>>>(b1, gb1, lr, 256);

    sgd_update_kernel<<<(128*256*3*3+255)/256,256>>>(w2, gw2, lr, 128*256*3*3);
    sgd_update_kernel<<<(128+255)/256,256>>>(b2, gb2, lr, 128);

    sgd_update_kernel<<<(128*128*3*3+255)/256,256>>>(w3, gw3, lr, 128*128*3*3);
    sgd_update_kernel<<<(128+255)/256,256>>>(b3, gb3, lr, 128);

    sgd_update_kernel<<<(256*128*3*3+255)/256,256>>>(w4, gw4, lr, 256*128*3*3);
    sgd_update_kernel<<<(256+255)/256,256>>>(b4, gb4, lr, 256);

    sgd_update_kernel<<<(3*256*3*3+255)/256,256>>>(w5, gw5, lr, 3*256*3*3);
    sgd_update_kernel<<<(3+255)/256,256>>>(b5, gb5, lr, 3);
}
