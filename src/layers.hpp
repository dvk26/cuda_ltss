#pragma once
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include "tensor.hpp"

// -------------------- Conv2D(3x3, pad=1, stride=1) --------------------

class Conv2D {
private:
    int inC_, outC_;
    std::vector<float> W_, b_, gW_, gb_;

    const Tensor* x_;     // lưu pointer, không copy
    Tensor out_;          // output buffer
    Tensor dX_;           // gradient input

public:
    Conv2D(int inC, int outC)
        : inC_(inC), outC_(outC),
          W_((size_t)outC*inC*9),
          b_((size_t)outC,0.f),
          gW_(W_.size(),0.f),
          gb_(b_.size(),0.f)
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> nd(0.f, std::sqrt(2.f/(inC*9)));
        for (auto& w : W_) w = nd(rng);
    }

    const Tensor& forward(const Tensor& x){
        x_ = &x;
        out_.resize(x.N(), outC_, x.H(), x.W());

        for(int n=0;n<x.N();++n)
        for(int oc=0;oc<outC_;++oc)
        for(int h=0;h<x.H();++h)
        for(int w=0;w<x.W();++w){
            float sum = b_[oc];
            for(int ic=0;ic<inC_;++ic)
            for(int kh=-1;kh<=1;++kh)
            for(int kw=-1;kw<=1;++kw){
                int ih=h+kh, iw=w+kw;
                if(ih<0||ih>=x.H()||iw<0||iw>=x.W()) continue;

                int k=(kh+1)*3+(kw+1);
                size_t widx = (((size_t)oc*inC_)+ic)*9 + k;
                sum += x.at(n,ic,ih,iw) * W_[widx];
            }
            out_.at(n,oc,h,w)=sum;
        }
        return out_;
    }

    const Tensor& backward(const Tensor& dY){
        const Tensor& x = *x_;

        std::fill(gW_.begin(), gW_.end(), 0.f);
        std::fill(gb_.begin(), gb_.end(), 0.f);

        dX_.resize(x.N(), inC_, x.H(), x.W());

        for(int n=0;n<x.N();++n)
        for(int oc=0;oc<outC_;++oc)
        for(int h=0;h<x.H();++h)
        for(int w=0;w<x.W();++w)
        {
            float gout = dY.at(n,oc,h,w);

            for(int ic=0;ic<inC_;++ic)
            for(int kh=-1;kh<=1;++kh)
            for(int kw=-1;kw<=1;++kw){
                int ih=h+kh, iw=w+kw;
                if(ih<0||ih>=x.H()||iw<0||iw>=x.W()) continue;
                int k=(kh+1)*3+(kw+1);
                size_t widx = (((size_t)oc*inC_)+ic)*9 + k;

                gW_[widx] += x.at(n,ic,ih,iw) * gout;
                dX_.at(n,ic,ih,iw) += W_[widx] * gout;
            }
            
            gb_[oc] += gout;
        }
        return dX_;
    }

    void sgd(float lr){
        for(size_t i=0;i<W_.size();++i) W_[i] -= lr * gW_[i];
        for(size_t i=0;i<b_.size();++i) b_[i] -= lr * gb_[i];
    }

    // Trả về tham chiếu tới vector chứa trọng số
    const std::vector<float>& weights() const { return W_; }

    // Nếu muốn lưu cả bias, thêm hàm này:
    const std::vector<float>& bias() const { return b_; }
};

// -------------------- ReLU --------------------
class ReLU {
private:
    const Tensor* x_;
    Tensor out_, dX_;

public:
    const Tensor& forward(const Tensor& x){
        x_ = &x;
        out_.resize(x.N(), x.C(), x.H(), x.W());

        for(size_t i=0;i<x.raw().size();++i)
            out_.raw()[i] = std::max(0.f, x.raw()[i]);
        return out_;
    }

    const Tensor& backward(const Tensor& dY){
        const Tensor& x = *x_;
        dX_.resize(x.N(), x.C(), x.H(), x.W());

        for(size_t i=0;i<x.raw().size();++i)
            dX_.raw()[i] = (x.raw()[i] > 0 ? dY.raw()[i] : 0.f);
        return dX_;
    }
};


// -------------------- MaxPool 2x2 (stride=2) --------------------
class MaxPool2x2 {
private:
    std::vector<int> idx_;   // argmax (0..3) cho mỗi output
    int Hout_ = 0, Wout_ = 0;

    const Tensor* x_ = nullptr; // trỏ vào input, không copy
    Tensor out_;                // buffer output
    Tensor dX_;                 // buffer gradient input

public:
    const Tensor& forward(const Tensor& x) {
        x_ = &x;
        Hout_ = x.H() / 2;
        Wout_ = x.W() / 2;

        out_.resize(x.N(), x.C(), Hout_, Wout_);
        idx_.assign((size_t)x.N() * x.C() * Hout_ * Wout_, 0);

        size_t p = 0;
        for (int n = 0; n < x.N(); ++n)
        for (int c = 0; c < x.C(); ++c) {
            for (int h = 0; h < Hout_; ++h)
            for (int w = 0; w < Wout_; ++w, ++p) {
                float best = -1e9f;
                int bestk = 0;

                for (int kh = 0; kh < 2; ++kh)
                for (int kw = 0; kw < 2; ++kw) {
                    int ih = h * 2 + kh;
                    int iw = w * 2 + kw;
                    float v = x.at(n, c, ih, iw);
                    int k = kh * 2 + kw;
                    if (v > best) {
                        best  = v;
                        bestk = k;
                    }
                }

                out_.at(n, c, h, w) = best;
                idx_[p] = bestk;
            }
        }
        return out_;
    }

    const Tensor& backward(const Tensor& dY) {
        const Tensor& x = *x_;

        dX_.resize(x.N(), x.C(), x.H(), x.W());
        std::fill(dX_.raw().begin(), dX_.raw().end(), 0.f);

        size_t p = 0;
        for (int n = 0; n < x.N(); ++n)
        for (int c = 0; c < x.C(); ++c) {
            for (int h = 0; h < dY.H(); ++h)
            for (int w = 0; w < dY.W(); ++w, ++p) {
                int k  = idx_[p];
                int kh = k / 2;
                int kw = k % 2;
                int ih = h * 2 + kh;
                int iw = w * 2 + kw;

                dX_.at(n, c, ih, iw) += dY.at(n, c, h, w);
            }
        }
        return dX_;
    }
};

// -------------------- Upsample 2x (nearest) --------------------
class Upsample2x {
private:
    const Tensor* x_ = nullptr; // không cần nhiều cache
    Tensor out_;
    Tensor dX_;

public:
    const Tensor& forward(const Tensor& x) {
        x_ = &x;
        out_.resize(x.N(), x.C(), x.H() * 2, x.W() * 2);

        for (int n = 0; n < x.N(); ++n)
        for (int c = 0; c < x.C(); ++c)
        for (int h = 0; h < out_.H(); ++h)
        for (int w = 0; w < out_.W(); ++w) {
            int ih = h / 2;
            int iw = w / 2;
            out_.at(n, c, h, w) = x.at(n, c, ih, iw);
        }
        return out_;
    }

    const Tensor& backward(const Tensor& dY) {
        const Tensor& x = *x_;
        dX_.resize(x.N(), x.C(), x.H(), x.W());
        std::fill(dX_.raw().begin(), dX_.raw().end(), 0.f);

        for (int n = 0; n < x.N(); ++n)
        for (int c = 0; c < x.C(); ++c)
        for (int h = 0; h < dY.H(); ++h)
        for (int w = 0; w < dY.W(); ++w) {
            int ih = h / 2;
            int iw = w / 2;
            dX_.at(n, c, ih, iw) += dY.at(n, c, h, w);
        }
        return dX_;
    }
};



// -------------------- MSE Loss --------------------
// -------------------- MSE Loss --------------------
class MSELoss {
private:
    Tensor dX_;

public:
    // trả về (loss, dPred)
    std::pair<float, const Tensor&>
    forward_backward(const Tensor& pred, const Tensor& target) {
        // đảm bảo cùng size
        assert(pred.raw().size() == target.raw().size());

        dX_.resize(pred.N(), pred.C(), pred.H(), pred.W());

        const auto& p = pred.raw();
        const auto& t = target.raw();
        auto& g = dX_.raw();

        const size_t N = p.size(); //batch size
        float s = 0.f;

        for (size_t i = 0; i < N; ++i) {
            float diff = p[i] - t[i];
            s += diff * diff;
            g[i] = 2.f * diff / (float)N;
        }

        float loss = s / (float)N;
        return { loss, dX_ };
    }
};
