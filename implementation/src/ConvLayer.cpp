#include "yolo/ConvLayer.hpp"
#include <cmath>
#include <random>

ConvLayer::ConvLayer(int in_channels, int out_channels, int kernel_size, int stride, int padding)
    : inC_(in_channels), outC_(out_channels), kernel_size_(kernel_size),
      stride_(stride), padding_(padding) {

    int fan_in = inC_ * kernel_size_ * kernel_size_;
    float stddev = std::sqrt(2.0f / fan_in);
    std::mt19937 gen(0);
    std::normal_distribution<float> dist(0.0f, stddev);

    biases_.resize(out_channels, 0.0f);
    weights_.resize(out_channels * in_channels * kernel_size * kernel_size);
    for (float &w : weights_) w = dist(gen);
}

std::vector<float> ConvLayer::forward(const std::vector<float>& input, Shape in, Shape& out) {
    int outH = (in.H + 2 * padding_ - kernel_size_) / stride_ + 1;
    int outW = (in.W + 2 * padding_ - kernel_size_) / stride_ + 1;
    out = { outC_, outH, outW };

    std::vector<float> output(outC_ * outH * outW, 0.0f);

    auto idx_in = [&](int c, int h, int w){ return c * in.H * in.W + h * in.W + w; };
    auto idx_out = [&](int c, int h, int w){ return c * outH * outW + h * outW + w; };
    auto idx_w = [&](int c, int ic, int kh, int kw){ 
        return c * inC_ * kernel_size_ * kernel_size_ 
            + ic * kernel_size_ * kernel_size_ 
            + kh * kernel_size_ + kw; 
    };

    for (int c = 0; c < outC_; c++) {
        for (int h = 0; h < outH; h++) {
            for (int w = 0; w < outW; w++) {
                float sum = 0.0f;

                for (int ic = 0; ic < inC_; ic++) {
                    for (int kh = 0; kh < kernel_size_; kh++) {
                        for (int kw = 0; kw < kernel_size_; kw++) {
                            int inputH = h * stride_ - padding_ + kh;
                            int inputW = w * stride_ - padding_ + kw;

                            if (inputH >= 0 && inputH < in.H && 
                                inputW >= 0 && inputW < in.W) {
                                sum += input[idx_in(ic, inputH, inputW)] 
                                        * weights_[idx_w(c, ic, kh, kw)];
                            }
                        }
                    }
                }

                output[idx_out(c, h, w)] = leakyReLU(sum + biases_[c]);
            }
        }
    }

    return output;
}
