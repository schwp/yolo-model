#pragma once
#include "yolo/Layer.hpp"
#include <vector>

class ConvLayer : public Layer {
public:
    ConvLayer(int in_channels, int out_channels, int kernel_size, int stride, int padding);

    std::vector<float> forward(const std::vector<float>& input, Shape in, Shape& out) override;
    // std::vector<float> backward(const std::vector<float>& grad_output);

    // These methods are for testing purposes
    void setAlpha(float alpha) { alpha_ = alpha; }
    void setWeights(const std::vector<float>& weights) { weights_ = weights; }

private:
    int inC_, outC_, kernel_size_, stride_, padding_;
    float alpha_ = 0.1f;
    std::vector<float> weights_;
    std::vector<float> biases_;

    inline float leakyReLU(float x) const { return x < 0 ? alpha_ * x : x; }
};
