#include "yolo/FullyConnectedLayer.hpp"
#include <cassert>
#include <random>

FullyConnectedLayer::FullyConnectedLayer(int in_features, int out_features)
    : in_features_(in_features), out_features_(out_features) {
    std::mt19937 gen(0);
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in_features_));
    
    biases_.resize(out_features_, 0.0f);
    weights_.resize(in_features_ * out_features_);
    for (auto& w : weights_) w = dist(gen);
}

std::vector<float> FullyConnectedLayer::forward(const std::vector<float>& input, Shape in, Shape& out) {
    assert(input.size() == in_features_ && "Input size must match in_features");
    assert(input.size() == in.H * in.W && "Input size must match Shape dimensions");

    std::vector<float> output(out_features_);
    out = { 1, 1, out_features_ };

    for (int i = 0; i < out_features_; ++i) {
        float sum = biases_[i];
        int base = i * in_features_;

        for (int j = 0; j < in_features_; ++j) {
            sum += weights_[base + j] * input[j];
        }

        output[i] = leakyReLU(sum);
    }

    return output;
}
