#include "../include/dataset.hpp"
#include "gpu_autoencoder.hpp"
#include "svm.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono> 
#include <iomanip> 

// Cấu hình
const int LATENT_DIM = 128 * 8 * 8; // 8192 features
const int BATCH_SIZE = 64;

// Helper struct
struct SVMData {
    svm_problem prob;
    std::vector<svm_node*> x_space;
    std::vector<double> y_space;
    std::vector<svm_node> data_pool;
};

// Hàm trích xuất đặc trưng
void extract_features(GPUAutoencoder& ae, DataLoader& loader, 
                      std::vector<std::vector<float>>& out_features, 
                      std::vector<int>& out_labels,
                      const std::string& phase_name) {
    loader.reset(false);
    int count = 0;
    
    std::cout << "[" << phase_name << "] Start extracting features..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    while (loader.has_next()) {
        auto batch = loader.next();
        Tensor& img = batch.images; 
        
        if (img.N() != BATCH_SIZE) continue; 

        Tensor latent = ae.encode(img); 
        const float* raw_data = latent.raw().data();

        for (int i = 0; i < BATCH_SIZE; ++i) {
            std::vector<float> vec(LATENT_DIM);
            std::copy(raw_data + i * LATENT_DIM, 
                      raw_data + (i + 1) * LATENT_DIM, 
                      vec.begin());
            
            out_features.push_back(vec);
            out_labels.push_back(batch.labels[i]);
        }
        
        count += BATCH_SIZE;
        if (count % 5000 == 0) std::cout << "  Processed " << count << " images...\r" << std::flush;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << "\n[" << phase_name << "] Done. Count: " << out_features.size() 
              << " | Time: " << std::fixed << std::setprecision(2) << elapsed.count() << "s" << std::endl;
}

SVMData prepare_libsvm_data(const std::vector<std::vector<float>>& features, 
                            const std::vector<int>& labels) {
    SVMData dataset;
    int N = features.size();
    dataset.prob.l = N;
    dataset.y_space.resize(N);
    dataset.x_space.resize(N);

    // Cấp phát bộ nhớ cho toàn bộ các node
    size_t total_nodes = (size_t)N * (LATENT_DIM + 1);
    dataset.data_pool.resize(total_nodes);

    size_t pool_idx = 0;
    for (int i = 0; i < N; ++i) {
        dataset.y_space[i] = labels[i];
        dataset.x_space[i] = &dataset.data_pool[pool_idx];

        for (int j = 0; j < LATENT_DIM; ++j) {
            dataset.data_pool[pool_idx].index = j + 1;
            dataset.data_pool[pool_idx].value = features[i][j];
            pool_idx++;
        }
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

    auto total_start = std::chrono::high_resolution_clock::now();

    // 1. Load Dataset
    std::cout << "Loading dataset..." << std::endl;
    CIFAR10 ds;
    ds.load(cifar_dir, false);
    DataLoader train_loader(ds.train_images(), ds.train_labels(), BATCH_SIZE, false);
    DataLoader test_loader(ds.test_images(), ds.test_labels(), BATCH_SIZE, false);

    // 2. Load AE
    GPUAutoencoder ae(BATCH_SIZE, 32, 32);
    try {
        ae.load_weights(weights_path);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    // ---------------------------------------------------------
    // 3. Extract Train Features
    // ---------------------------------------------------------
    std::vector<std::vector<float>> train_feats;
    std::vector<int> train_labels;
    extract_features(ae, train_loader, train_feats, train_labels, "TRAIN-SET");

    // ---------------------------------------------------------
    // 4. Min-Max Scaling Logic
    // ---------------------------------------------------------
    std::cout << "Computing Scaling Parameters (Min-Max)..." << std::endl;
    std::vector<float> max_val(LATENT_DIM, -1e9), min_val(LATENT_DIM, 1e9);

    // Tính min/max dựa trên tập TRAIN
    for(const auto& vec : train_feats) {
        for(int j=0; j<LATENT_DIM; ++j) {
            if(vec[j] > max_val[j]) max_val[j] = vec[j];
            if(vec[j] < min_val[j]) min_val[j] = vec[j];
        }
    }

    // Lambda hàm scale
    auto scale_data = [&](std::vector<std::vector<float>>& data) {
        for(auto& vec : data) {
            for(int j=0; j<LATENT_DIM; ++j) {
                float range = max_val[j] - min_val[j];
                if(range > 1e-6) 
                    vec[j] = (vec[j] - min_val[j]) / range; // Scale về [0, 1]
                else 
                    vec[j] = 0.0f;
            }
        }
    };

    // Áp dụng scale cho TRAIN ngay lập tức
    scale_data(train_feats);

    // ---------------------------------------------------------
    // 5. Extract & Scale Test Features
    // ---------------------------------------------------------
    std::vector<std::vector<float>> test_feats;
    std::vector<int> test_labels;
    // Extract xong mới được scale
    extract_features(ae, test_loader, test_feats, test_labels, "TEST-SET");
    std::cout << "Scaling Test Data using Train parameters..." << std::endl;
    scale_data(test_feats); // [QUAN TRỌNG] Scale tập test bằng min/max của train

    // ---------------------------------------------------------
    // 6. Train SVM
    // ---------------------------------------------------------
    std::cout << "Converting to LIBSVM sparse format..." << std::endl;
    auto t_prep_start = std::chrono::high_resolution_clock::now();
    SVMData train_data = prepare_libsvm_data(train_feats, train_labels);
    std::cout << "Conversion time: " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_prep_start).count() << "s\n";

    svm_parameter param;
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 3;
    param.gamma = 1.0 / LATENT_DIM; 
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 2000; // 2GB Cache
    param.C = 10; 
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;

    std::cout << "Training SVM (RBF Kernel)..." << std::endl;
    auto t_train_start = std::chrono::high_resolution_clock::now();
    
    svm_model* model = svm_train(&train_data.prob, &param);
    
    std::cout << "SVM Training time: " << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_train_start).count() << "s\n";


    const char* model_filename = "cifar10_svm.model";
    if (svm_save_model(model_filename, model) == 0) {
        std::cout << "Saved SVM model to " << model_filename << "\n";
    } else {
        std::cerr << "Failed to save SVM model.\n";
    }

    

    // ---------------------------------------------------------
    // 7. Evaluate
    // ---------------------------------------------------------
    std::cout << "Evaluating accuracy..." << std::endl;
    auto t_eval_start = std::chrono::high_resolution_clock::now();
    
    int correct = 0;
    int total = test_feats.size();
    std::vector<svm_node> test_nodes(LATENT_DIM + 1);

    for (int i = 0; i < total; ++i) {
        for (int j = 0; j < LATENT_DIM; ++j) {
            test_nodes[j].index = j + 1;
            test_nodes[j].value = test_feats[i][j];
        }
        test_nodes[LATENT_DIM].index = -1;

        double pred = svm_predict(model, test_nodes.data());
        if ((int)pred == test_labels[i]) correct++;
    }

    float acc = 100.0f * correct / total;
    auto total_end = std::chrono::high_resolution_clock::now();

    std::cout << "------------------------------------------------\n";
    std::cout << "Total Runtime:   " << std::chrono::duration<double>(total_end - total_start).count() << "s\n";
    std::cout << "TEST ACCURACY:   " << acc << "%\n";
    std::cout << "------------------------------------------------\n";

    svm_free_and_destroy_model(&model);
    return 0;
}