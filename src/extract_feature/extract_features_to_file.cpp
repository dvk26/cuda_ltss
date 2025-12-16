#include "../include/dataset.hpp"
#include "gpu_autoencoder.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream> // Thư viện để ghi file
#include <chrono> 
#include <iomanip>

// Cấu hình
const int LATENT_DIM = 128 * 8 * 8; // 8192 features
const int BATCH_SIZE = 64;

// Hàm Extract và Lưu xuống file Binary
// Cấu trúc file: [Header: NumSamples(int), Dim(int)] + [Data: Label(int), FeatureVector(float array)...]
void extract_and_save(GPUAutoencoder& ae, DataLoader& loader, const std::string& filename) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << filename << "\n";
        return;
    }

    std::cout << "Saving to " << filename << "..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // 1. Ghi Header tạm (Số lượng mẫu = 0, Số chiều = 8192)
    // Sau khi chạy xong sẽ quay lại (seek) để cập nhật số lượng mẫu chính xác
    int total_samples = 0;
    int dim = LATENT_DIM;
    outfile.write(reinterpret_cast<const char*>(&total_samples), sizeof(int));
    outfile.write(reinterpret_cast<const char*>(&dim), sizeof(int));

    loader.reset(false);
    
    while (loader.has_next()) {
        auto batch = loader.next();
        Tensor& img = batch.images;

        // Bỏ qua batch lẻ không đủ size (do GPU code fix cứng size)
        if (img.N() != BATCH_SIZE) continue;

                // --- GPU Inference ---
        Tensor latent = ae.encode(img);  // encode() đã copy về CPU rồi
        
        // latent.raw().data() là con trỏ CPU, dùng trực tiếp luôn
        const float* latent_data = latent.raw().data();
        
        // --- Ghi xuống file ---
        for (int i = 0; i < BATCH_SIZE; ++i) {
            // 1. Ghi label
            int label = batch.labels[i];
            outfile.write(reinterpret_cast<const char*>(&label), sizeof(int));
        
            // 2. Ghi feature vector
            const float* feature_ptr = latent_data + i * LATENT_DIM;
            outfile.write(reinterpret_cast<const char*>(feature_ptr),
                          LATENT_DIM * sizeof(float));
        }

        total_samples += BATCH_SIZE;
        if (total_samples % 5000 == 0) {
            std::cout << "  Extracted " << total_samples << " samples...\r" << std::flush;
        }
    }

    // Quay lại đầu file để cập nhật tổng số mẫu thực tế
    outfile.seekp(0);
    outfile.write(reinterpret_cast<const char*>(&total_samples), sizeof(int));
    outfile.close();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "\nDone! Saved " << total_samples << " samples to " << filename 
              << " (" << std::fixed << std::setprecision(2) << elapsed.count() << "s)\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./extract_step2 <cifar_dir> <weights_path.bin>\n";
        return 1;
    }
    std::string cifar_dir = argv[1];
    std::string weights_path = argv[2];

    // 1. Load Dataset
    std::cout << "Loading dataset..." << std::endl;
    CIFAR10 ds;
    ds.load(cifar_dir, false);
    DataLoader train_loader(ds.train_images(), ds.train_labels(), BATCH_SIZE, false);
    DataLoader test_loader(ds.test_images(), ds.test_labels(), BATCH_SIZE, false);

    // 2. Load Autoencoder
    GPUAutoencoder ae(BATCH_SIZE, 32, 32);
    try {
        ae.load_weights(weights_path);
        std::cout << "Weights loaded successfully.\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    // 3. Extract & Save Train Set
    extract_and_save(ae, train_loader, "train_features.bin");

    // 4. Extract & Save Test Set
    extract_and_save(ae, test_loader, "test_features.bin");

    std::cout << "\nAll features extracted successfully. Ready for Python/cuML!" << std::endl;
    return 0;
}
