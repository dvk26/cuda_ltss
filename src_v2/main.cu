#include <iostream>
#include "dataset.hpp"
#include "gpu_net.hpp"

int main(int argc, char** argv) {
    std::string cifar_dir = argv[1];
    CIFAR10 ds;
    ds.load(cifar_dir, true);

    int batch_size = 64;
    GPUNet net(batch_size);

    DataLoader train_loader(ds.train_images(), ds.train_labels(), batch_size, true);

    int epochs = 5;
    float lr = 1e-3;

    for (int ep = 1; ep <= epochs; ++ep) {
        train_loader.reset(true);
        double epoch_loss = 0.0;
        int nb = 0;

        while (train_loader.has_next()) {
            auto batch = train_loader.next();
            int N = batch.size;

            size_t bytes = (size_t)N * 3 * 32 * 32 * sizeof(float);
            CUDA_CHECK(cudaMemcpy(net.x, batch.images.raw().data(), bytes, cudaMemcpyHostToDevice));

            net.forward(N);

            float* d_target;
            CUDA_CHECK(cudaMalloc(&d_target, bytes));
            CUDA_CHECK(cudaMemcpy(d_target, batch.images.raw().data(), bytes, cudaMemcpyHostToDevice));

            float loss = net.loss(d_target, N);
            CUDA_CHECK(cudaFree(d_target));

            net.backward(lr, N);

            epoch_loss += loss;
            ++nb;
        }

        std::cout << "Epoch " << ep << " loss = " << (epoch_loss / nb) << "\n";
    }

    return 0;
}