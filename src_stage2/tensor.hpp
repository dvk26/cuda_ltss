#pragma once
#include <vector>
#include <cassert>
#include <algorithm>

class Tensor {
private:
    int N_, C_, H_, W_;
    std::vector<float> data_;

public:
    Tensor(): N_(0),C_(0),H_(0),W_(0) {}
    Tensor(int n,int c,int h,int w)
        : N_(n),C_(c),H_(h),W_(w),data_((size_t)n*c*h*w,0.f) {}

    inline float& at(int n,int c,int h,int w) {
        return data_[ ((size_t)n*C_ + c)*H_*W_ + (size_t)h*W_ + w ];
    }
    inline const float& at(int n,int c,int h,int w) const {
        return data_[ ((size_t)n*C_ + c)*H_*W_ + (size_t)h*W_ + w ];
    }

    int N() const { return N_; }
    int C() const { return C_; }
    int H() const { return H_; }
    int W() const { return W_; }

    std::vector<float>& raw() { return data_; }
    const std::vector<float>& raw() const { return data_; }

    void resize_like(const Tensor& other) {
        N_=other.N_; C_=other.C_; H_=other.H_; W_=other.W_;
        data_.assign(other.raw().size(), 0.f);
    }
    void resize(int n,int c,int h,int w) {
        N_ = n; C_ = c; H_ = h; W_ = w;
        data_.assign((size_t)n*c*h*w, 0.f);
    }
};
