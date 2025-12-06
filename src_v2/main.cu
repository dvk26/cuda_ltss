#include <iostream>
#include "dataset.hpp"
#include "gpu_net.hpp"

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}

int main(int argc, char** argv) {
    if (argc < 2 || argv[1] == nullptr) {
        std::cerr << "Usage: " << argv[0] << " <cifar10_data_dir>\n";
        return 1;
    }

    std::string cifar_dir = argv[1];
    CIFAR10 ds;
    ds.load(cifar_dir, true);

    int batch_size = 64;
    GPUNet net(batch_size);

    DataLoader train_loader(
        ds.train_images(),
        ds.train_labels(),
        batch_size,
        true
    );

    int epochs = 5;
    float lr = 1e-3f;

    // Buffer target max-size
    size_t max_bytes = (size_t)batch_size * 3 * 32 * 32 * sizeof(float);
    float* d_target = nullptr;
    CUDA_CHECK(cudaMalloc(&d_target, max_bytes));

    for (int ep = 1; ep <= epochs; ++ep) {
        printf("\n===== EPOCH %d =====\n", ep);

        train_loader.reset(true);
        double epoch_loss = 0.0;
        int nb = 0;

        while (train_loader.has_next()) {
            auto batch = train_loader.next();
            int N = batch.size;

            // Bỏ batch thiếu để tránh mismatch với buffer GPU
            if (N != batch_size) {
                std::cout << "Skip batch (N=" << N << ")\n";
                continue;
            }

            size_t bytes = (size_t)N * 3 * 32 * 32 * sizeof(float);

            // Copy input
            CUDA_CHECK(cudaMemcpy(
                net.x,
                batch.images.raw().data(),
                bytes,
                cudaMemcpyHostToDevice
            ));

            // Copy target
            CUDA_CHECK(cudaMemcpy(
                d_target,
                batch.images.raw().data(),
                bytes,
                cudaMemcpyHostToDevice
            ));

            // -------- Forward --------
            net.forward(N);
            CUDA_CHECK(cudaDeviceSynchronize());

            // -------- Loss --------
            float loss = net.loss(d_target, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::cout << "Batch " << nb << " loss = " << loss << "\n";

            epoch_loss += loss;
            nb++;
        }

        if (nb > 0)
            std::cout << "Epoch " << ep << " loss = " << (epoch_loss / nb) << "\n";
        else
            std::cout << "Epoch " << ep << " loss = (no valid batch)\n";
    }

    CUDA_CHECK(cudaFree(d_target));
    return 0;
}
