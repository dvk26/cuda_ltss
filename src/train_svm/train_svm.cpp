#include "../include/dataset.hpp"
#include "gpu_autoencoder.hpp"
#include "svm.h" // File header của LIBSVM
#include <iostream>
#include <vector>
#include <algorithm>

// Cấu hình theo đề bài
const int LATENT_DIM = 128 * 8 * 8; // 8192 features
const int BATCH_SIZE = 64;

// Hàm helper để chuyển vector float sang định dạng svm_node của LIBSVM
struct SVMData {
    svm_problem prob;
    std::vector<svm_node*> x_space; // Con trỏ mảng feature cho từng mẫu
    std::vector<double> y_space;    // Label
    std::vector<svm_node> data_pool; // Bộ nhớ thực chứa dữ liệu
};

// Trích xuất đặc trưng từ Autoencoder
void extract_features(GPUAutoencoder& ae, DataLoader& loader, 
                      std::vector<std::vector<float>>& out_features, 
                      std::vector<int>& out_labels) {
    loader.reset(false); // Không shuffle để giữ thứ tự nếu cần kiểm tra
    int count = 0;
    
    std::cout << "Extracting features..." << std::endl;

    while (loader.has_next()) {
        auto batch = loader.next();
        Tensor& img = batch.images; // [N, 3, 32, 32]
        
        if (img.N() != BATCH_SIZE) continue; // Bỏ qua batch lẻ (do GPU code fix cứng size)

        // 1. Chạy Encode trên GPU
        Tensor latent = ae.encode(img); // [N, 128, 8, 8]

        // 2. Chép về CPU và Flatten
        const float* raw_data = latent.raw().data();
        for (int i = 0; i < BATCH_SIZE; ++i) {
            std::vector<float> vec(LATENT_DIM);
            // Copy 8192 float cho ảnh thứ i
            std::copy(raw_data + i * LATENT_DIM, 
                      raw_data + (i + 1) * LATENT_DIM, 
                      vec.begin());
            
            out_features.push_back(vec);
            out_labels.push_back(batch.labels[i]);
        }
        
        count += BATCH_SIZE;
        if (count % 1000 == 0) std::cout << "Processed " << count << " images...\r" << std::flush;
    }
    std::cout << "\nExtraction done. Total: " << out_features.size() << "\n";
}

// Chuẩn bị dữ liệu cho LIBSVM
SVMData prepare_libsvm_data(const std::vector<std::vector<float>>& features, 
                            const std::vector<int>& labels) {
    SVMData dataset;
    int N = features.size();
    dataset.prob.l = N;
    dataset.y_space.resize(N);
    dataset.x_space.resize(N);

    // LIBSVM dùng sparse format: index:value. Kết thúc bằng index = -1.
    // Vì feature dense (8192 chiều), ta cần (8192 + 1) node cho mỗi ảnh.
    // Tổng số node cần cấp phát: N * (LATENT_DIM + 1)
    size_t total_nodes = (size_t)N * (LATENT_DIM + 1);
    dataset.data_pool.resize(total_nodes);

    size_t pool_idx = 0;
    for (int i = 0; i < N; ++i) {
        dataset.y_space[i] = labels[i];
        dataset.x_space[i] = &dataset.data_pool[pool_idx];

        for (int j = 0; j < LATENT_DIM; ++j) {
            dataset.data_pool[pool_idx].index = j + 1; // LIBSVM index start from 1
            dataset.data_pool[pool_idx].value = features[i][j];
            pool_idx++;
        }
        // Node kết thúc
        dataset.data_pool[pool_idx].index = -1;
        pool_idx++;
    }

    dataset.prob.y = dataset.y_space.data();
    dataset.prob.x = dataset.x_space.data();
    return dataset;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./train_svm <cifar_dir> <weights_path.bin>\n";
        return 1;
    }
    std::string cifar_dir = argv[1];
    std::string weights_path = argv[2];

    // 1. Load Dataset
    CIFAR10 ds;
    ds.load(cifar_dir, false);
    DataLoader train_loader(ds.train_images(), ds.train_labels(), BATCH_SIZE, false);
    DataLoader test_loader(ds.test_images(), ds.test_labels(), BATCH_SIZE, false);

    // 2. Load Autoencoder & Weights
    GPUAutoencoder ae(BATCH_SIZE, 32, 32);
    try {
        ae.load_weights(weights_path);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    // 3. Extract Features (Train Set)
    std::vector<std::vector<float>> train_feats;
    std::vector<int> train_labels;
    extract_features(ae, train_loader, train_feats, train_labels);

    // 4. Prepare SVM Data
    std::cout << "Preparing SVM training data format...\n";
    SVMData train_data = prepare_libsvm_data(train_feats, train_labels);

    // 5. Setup SVM Parameters (Theo yêu cầu PDF)
    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = RBF;    // Radial Basis Function
    param.degree = 3;
    param.gamma = 1.0 / LATENT_DIM; // gamma = auto (1/num_features)
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 2000;    // MB RAM cho cache kernel (quan trọng để chạy nhanh)
    param.C = 10;               // C = 10 theo đề bài
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;

    // 6. Train SVM
    std::cout << "Training SVM (This may take a while)..." << std::endl;
    // Tắt log của libsvm để đỡ rối (optional)
    // svm_set_print_string_function(NULL); 
    
    svm_model* model = svm_train(&train_data.prob, &param);
    std::cout << "SVM Training Completed.\n";

    // 7. Evaluate on Test Set
    std::cout << "Extracting Test Features...\n";
    std::vector<std::vector<float>> test_feats;
    std::vector<int> test_labels;
    extract_features(ae, test_loader, test_feats, test_labels);

    std::cout << "Evaluating...\n";
    int correct = 0;
    int total = test_feats.size();
    
    // Buffer tạm để dự đoán từng mẫu
    std::vector<svm_node> test_nodes(LATENT_DIM + 1);

    for (int i = 0; i < total; ++i) {
        // Convert single vector to svm_node
        for (int j = 0; j < LATENT_DIM; ++j) {
            test_nodes[j].index = j + 1;
            test_nodes[j].value = test_feats[i][j];
        }
        test_nodes[LATENT_DIM].index = -1;

        double pred = svm_predict(model, test_nodes.data());
        if ((int)pred == test_labels[i]) {
            correct++;
        }
    }

    float acc = 100.0f * correct / total;
    std::cout << "Test Accuracy: " << acc << "% (Expected 60-65%)\n";

    // Cleanup
    svm_free_and_destroy_model(&model);
    
    return 0;
}