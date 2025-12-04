#pragma once
#include <cuda_runtime.h>

struct GPUNet {
    // ===== Weights, bias, grads =====
    float *w1, *b1, *gw1, *gb1;
    float *w2, *b2, *gw2, *gb2;
    float *w3, *b3, *gw3, *gb3;
    float *w4, *b4, *gw4, *gb4;
    float *w5, *b5, *gw5, *gb5;

    // ===== Activations =====
    float *x;
    float *a1, *a1_relu, *p1;
    float *a2, *a2_relu, *p2;
    float *a3, *a3_relu, *u1;
    float *a4, *a4_relu, *u2;
    float *out;

    // ===== Gradients =====
    float *g_x;
    float *g_a1, *g_a1_relu, *g_p1;
    float *g_a2, *g_a2_relu, *g_p2;
    float *g_a3, *g_a3_relu, *g_u1;
    float *g_a4, *g_a4_relu, *g_u2;
    float *g_out;

    // ===== Pool indices =====
    int *idx_p1, *idx_p2;

    // ===== MSE buffers =====
    float *mse_buf, *mse_dPred, *mse_loss_dev;

    int Nmax;

    GPUNet(int Nmax_);
    ~GPUNet();

    void forward(int N);
    float loss(const float* d_target, int N);
    void backward(float lr, int N);
};