#pragma once
#include <cuda_runtime.h>

struct GPUNet {
    int Nmax;

    // Device pointers
    float *x, *a1, *p1, *a2, *p2, *a3, *u1, *a4, *u2, *out;
    float *w1, *b1, *gw1, *gb1;
    float *w2, *b2, *gw2, *gb2;
    float *w3, *b3, *gw3, *gb3;
    float *w4, *b4, *gw4, *gb4;
    float *w5, *b5, *gw5, *gb5;
    int *idx_p1, *idx_p2;
    float *mse_buf, *mse_dPred, *mse_loss_dev;

    GPUNet(int Nmax_);
    ~GPUNet();

    void forward(int N);
    float loss(const float* d_target, int N);
    void backward(float lr, int N);
};