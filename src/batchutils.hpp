// batch_utils.hpp (tuỳ bạn đặt tên)
#pragma once
#include "tensor.hpp"

// Lấy batch ảnh [B, C, H, W] từ tensor all [N, C, H, W] bắt đầu từ start
inline Tensor get_batch(const Tensor& all, int start, int batch_size) {
    int N = all.N();
    int C = all.C();
    int H = all.H();
    int W = all.W();

    // đảm bảo không vượt N
    int B = std::min(batch_size, N - start);

    Tensor out(B, C, H, W);
    for (int b = 0; b < B; ++b) {
        int n = start + b;
        for (int c = 0; c < C; ++c)
        for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
            out.at(b, c, h, w) = all.at(n, c, h, w);
    }
    return out;
}
