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

    DataLoader train_loader(ds.train_images(), ds.train_labels(), batch_size, true);

    int epochs = 5;
    float lr = 1e-3;

    size_t max_bytes = (size_t)batch_size * 3 * 32 * 32 * sizeof(float);
    float* d_target = nullptr;
    CUDA_CHECK(cudaMalloc(&d_target, max_bytes));

    for (int ep = 1; ep <= epochs; ++ep) {
        train_loader.reset(true);
        double epoch_loss = 0.0;
        int nb = 0;

        while (train_loader.has_next()) {
            auto batch = train_loader.next();
            int N = batch.size;

            // ⚠️ Bỏ batch cuối nếu N < batch_size
            if (N != batch_size) {
                std::cout << "Skip last batch with N = " << N
                          << " (batch_size = " << batch_size << ")\n";
                break;
            }

            std::cout << "N=" << N << " total=" << (N*3*32*32) << std::endl;
            std::cout << "out=" << net.out << " d_target=" << d_target << " mse_buf=" << net.mse_buf << std::endl;

            size_t bytes = (size_t)N * 3 * 32 * 32 * sizeof(float);
            CUDA_CHECK(cudaMemcpy(net.x, batch.images.raw().data(), bytes, cudaMemcpyHostToDevice));

            // ---- forward ----
            net.forward(N);
            CUDA_CHECK(cudaDeviceSynchronize());

            // ---- target ----
            CUDA_CHECK(cudaMemcpy(d_target, batch.images.raw().data(), bytes, cudaMemcpyHostToDevice));

            // ---- loss ----
            float loss = net.loss(d_target, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            // ---- backward ----
            net.backward(lr, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            epoch_loss += loss;
            ++nb;
        }

        std::cout << "Epoch " << ep << " loss = " << (epoch_loss / nb) << "\n";
    }

    CUDA_CHECK(cudaFree(d_target));
    return 0;
}

__global__ void mse_elementwise_kernel(const float* pred, const float* target, float* buf, float* dPred, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float diff = pred[idx] - target[idx];
        buf[idx] = diff * diff;
        dPred[idx] = 2.0f * diff / total;
    }
}

float GPUNet::loss(const float* d_target, int N) {
    int total = N * 3 * 32 * 32;
    return mse_forward_backward(
        out, d_target,
        mse_buf, mse_dPred,
        mse_loss_dev, total
    );
}