#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include "tensor.hpp"

class CIFAR10 {
private:
    Tensor images_;                 // [N,3,32,32] in [0,1]
    std::vector<uint8_t> labels_;   // [N]

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

    void load(const std::string& dir) {
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

        images_ = Tensor((int)(n_train+n_test), C, H, W);
        labels_.assign(n_train+n_test, 0);

        std::vector<uint8_t> raw;
        size_t off=0;
        auto load_one = [&](const std::string& file){
            read_file(file, raw);
            size_t recs = raw.size()/REC;
            for(size_t i=0;i<recs;i++){
                size_t base=i*REC;
                uint8_t lbl=raw[base];
                labels_[off+i]=lbl;
                const uint8_t* pix = raw.data()+base+1;
                for(int h=0;h<H;h++) for(int w=0;w<W;w++){
                    int idx = h*W + w;
                    images_.at((int)(off+i),0,h,w) = pix[idx] / 255.f;
                    images_.at((int)(off+i),1,h,w) = pix[1024+idx] / 255.f;
                    images_.at((int)(off+i),2,h,w) = pix[2048+idx] / 255.f;
                }
            }
            off += recs;
        };

        for (auto* f: trains) load_one(dir + "/" + f);
        load_one(dir + "/test_batch.bin");
    }

    const Tensor& images() const { return images_; }
    const std::vector<uint8_t>& labels() const { return labels_; }

    // by CIFAR convention:
    int train_size() const { return 50000; }
    int test_size()  const { return 10000; }
};
