#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <random>
#include <numeric>
#include "tensor.hpp"

class CIFAR10 {
private:
    Tensor train_images_;                 // [50000,3,32,32] in [0,1]
    Tensor test_images_;                  // [10000,3,32,32] in [0,1]
    std::vector<uint8_t> train_labels_;   // [50000]
    std::vector<uint8_t> test_labels_;    // [10000]
    bool keep_10_percent_;                // true: keep 10%, false: keep all

    static void read_file(const std::string& path, std::vector<uint8_t>& buf) {
        std::ifstream f(path, std::ios::binary);
        if(!f) throw std::runtime_error("Cannot open " + path);
        f.seekg(0,std::ios::end);
        size_t sz=(size_t)f.tellg();
        f.seekg(0,std::ios::beg);
        buf.resize(sz);
        f.read((char*)buf.data(), sz);
    }

public:
    CIFAR10() = default;

    void load(const std::string& dir, bool keep_10_percent=true) {
        keep_10_percent_ = keep_10_percent;
        const int W=32,H=32,C=3, REC=1+W*H*C;
        const char* trains[5] = {
            "data_batch_1.bin","data_batch_2.bin","data_batch_3.bin",
            "data_batch_4.bin","data_batch_5.bin"
        };

        // count train/test
        size_t n_train=0;
        for (auto* f: trains) {
            std::ifstream fi(dir + "/" + f, std::ios::binary);
            if(!fi) throw std::runtime_error("Missing "+std::string(f));
            fi.seekg(0,std::ios::end);
            n_train += ((size_t)fi.tellg()) / REC;
        }
        std::ifstream te(dir + "/test_batch.bin", std::ios::binary);
        if(!te) throw std::runtime_error("Missing test_batch.bin");
        te.seekg(0,std::ios::end);
        size_t n_test = ((size_t)te.tellg()) / REC;

        train_images_ = Tensor((int)n_train, C, H, W);
        test_images_ = Tensor((int)n_test, C, H, W);
        train_labels_.assign(n_train, 0);
        test_labels_.assign(n_test, 0);

        std::vector<uint8_t> raw;
        size_t off=0;
        auto load_train = [&](const std::string& file){
            read_file(file, raw);
            size_t recs = raw.size()/REC;
            for(size_t i=0;i<recs;i++){
                size_t base=i*REC;
                uint8_t lbl=raw[base];
                train_labels_[off+i]=lbl;
                const uint8_t* pix = raw.data()+base+1;
                for(int h=0;h<H;h++) for(int w=0;w<W;w++){
                    int idx = h*W + w;
                    train_images_.at((int)(off+i),0,h,w) = pix[idx] / 255.f;
                    train_images_.at((int)(off+i),1,h,w) = pix[1024+idx] / 255.f;
                    train_images_.at((int)(off+i),2,h,w) = pix[2048+idx] / 255.f;
                }
            }
            off += recs;
        };

        off = 0;
        for (auto* f: trains) load_train(dir + "/" + f);
        
        off = 0;
        read_file(dir + "/test_batch.bin", raw);
        size_t recs = raw.size()/REC;
        for(size_t i=0;i<recs;i++){
            size_t base=i*REC;
            uint8_t lbl=raw[base];
            test_labels_[i]=lbl;
            const uint8_t* pix = raw.data()+base+1;
            for(int h=0;h<H;h++) for(int w=0;w<W;w++){
                int idx = h*W + w;
                test_images_.at((int)i,0,h,w) = pix[idx] / 255.f;
                test_images_.at((int)i,1,h,w) = pix[1024+idx] / 255.f;
                test_images_.at((int)i,2,h,w) = pix[2048+idx] / 255.f;
            }
        }

        // Chỉ giữ lại 5000 ảnh train và 1000 ảnh test (10% of original)
        if (keep_10_percent_) {
            if (train_images_.N() > 5000) {
                Tensor tmp(5000, 3, 32, 32);
                for (int i = 0; i < 5000; ++i)
                    for (int c = 0; c < 3; ++c)
                    for (int h = 0; h < 32; ++h)
                    for (int w = 0; w < 32; ++w)
                        tmp.at(i, c, h, w) = train_images_.at(i, c, h, w);
                train_images_ = std::move(tmp);
                train_labels_.resize(5000);
            }
            if (test_images_.N() > 1000) {
                Tensor tmp(1000, 3, 32, 32);
                for (int i = 0; i < 1000; ++i)
                    for (int c = 0; c < 3; ++c)
                    for (int h = 0; h < 32; ++h)
                    for (int w = 0; w < 32; ++w)
                        tmp.at(i, c, h, w) = test_images_.at(i, c, h, w);
                test_images_ = std::move(tmp);
                test_labels_.resize(1000);
            }
        }
    }

    const Tensor& train_images() const { return train_images_; }
    const Tensor& test_images() const { return test_images_; }
    const std::vector<uint8_t>& train_labels() const { return train_labels_; }
    const std::vector<uint8_t>& test_labels() const { return test_labels_; }

    // by CIFAR convention (or actual size if keep_10_percent is false):
    int train_size() const { return train_images_.N(); }
    int test_size()  const { return test_images_.N(); }
};

// -------------------- DataLoader --------------------
class DataLoader {
public:
    struct Batch {
        Tensor images;
        std::vector<uint8_t> labels;
        int size;
    };

private:
    const Tensor& images_;
    const std::vector<uint8_t>& labels_;
    int batch_size_;
    bool shuffle_;
    int current_idx_;
    int total_;
    std::vector<int> indices_;

public:
    DataLoader(const Tensor& images, const std::vector<uint8_t>& labels,
               int batch_size, bool shuffle = false)
        : images_(images), labels_(labels), batch_size_(batch_size),
          shuffle_(shuffle), current_idx_(0), total_(images.N())
    {
        indices_.resize(total_);
        std::iota(indices_.begin(), indices_.end(), 0);
        
        if (shuffle_) {
            std::mt19937 rng(42);
            std::shuffle(indices_.begin(), indices_.end(), rng);
        }
    }

    bool has_next() const {
        return current_idx_ < total_;
    }

    Batch next() {
        if (!has_next()) throw std::runtime_error("No more batches");
        
        int remaining = total_ - current_idx_;
        int cur_batch_size = std::min(batch_size_, remaining);
        
        Batch batch;
        batch.images.resize(cur_batch_size, 3, 32, 32);
        batch.labels.assign(cur_batch_size, 0);
        batch.size = cur_batch_size;
        
        for (int k = 0; k < cur_batch_size; ++k) {
            int id = indices_[current_idx_ + k];
            batch.labels[k] = labels_[id];
            
            for (int c = 0; c < 3; ++c)
            for (int h = 0; h < 32; ++h)
            for (int w = 0; w < 32; ++w)
                batch.images.at(k, c, h, w) = images_.at(id, c, h, w);
        }
        
        current_idx_ += cur_batch_size;
        return batch;
    }

    void reset(bool shuffle = false) {
        current_idx_ = 0;
        std::iota(indices_.begin(), indices_.end(), 0);
        
        if (shuffle) {
            std::mt19937 rng(std::random_device{}());
            std::shuffle(indices_.begin(), indices_.end(), rng);
        }
    }

    int num_batches() const {
        return (total_ + batch_size_ - 1) / batch_size_;
    }
};
