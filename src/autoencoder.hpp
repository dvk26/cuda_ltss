#pragma once
#include "layers.hpp"

// Enc: Conv(3->256)+ReLU+Pool2  -> Conv(256->128)+ReLU+Pool2  -> latent (8x8x128)
// Dec: Conv(128->128)+ReLU -> Up2 -> Conv(128->256)+ReLU -> Up2 -> Conv(256->3)

class Autoencoder {
private:
    Conv2D c1_{3,256};
    ReLU   r1_;
    MaxPool2x2 p1_;

    Conv2D c2_{256,128};
    ReLU   r2_;
    MaxPool2x2 p2_;

    Conv2D c3_{128,128};
    ReLU   r3_;
    Upsample2x up1_;

    Conv2D c4_{128,256};
    ReLU   r4_;
    Upsample2x up2_;

    Conv2D c5_{256,3};

public:
    const Tensor& forward(const Tensor& x){
        const Tensor& a1 = c1_.forward(x);
        const Tensor& a2 = r1_.forward(a1);
        const Tensor& a3 = p1_.forward(a2);

        const Tensor& a4 = c2_.forward(a3);
        const Tensor& a5 = r2_.forward(a4);
        const Tensor& a6 = p2_.forward(a5);

        const Tensor& a7 = c3_.forward(a6);
        const Tensor& a8 = r3_.forward(a7);
        const Tensor& u1 = up1_.forward(a8);

        const Tensor& a9 = c4_.forward(u1);
        const Tensor& a10 = r4_.forward(a9);
        const Tensor& u2 = up2_.forward(a10);

        return c5_.forward(u2);
    }

    void backward_and_update(const Tensor& dOut, float lr){
        const Tensor& d5 = c5_.backward(dOut);
        c5_.sgd(lr);

        const Tensor& du2 = up2_.backward(d5);
        const Tensor& da10 = r4_.backward(du2);
        const Tensor& da9  = c4_.backward(da10);
        c4_.sgd(lr);

        const Tensor& du1 = up1_.backward(da9);
        const Tensor& da8 = r3_.backward(du1);
        const Tensor& da7 = c3_.backward(da8);
        c3_.sgd(lr);

        const Tensor& da6 = p2_.backward(da7);
        const Tensor& da5 = r2_.backward(da6);
        const Tensor& da4 = c2_.backward(da5);
        c2_.sgd(lr);

        const Tensor& da3 = p1_.backward(da4);
        const Tensor& da2 = r1_.backward(da3);
        const Tensor& da1 = c1_.backward(da2);
        c1_.sgd(lr);
    }
};
