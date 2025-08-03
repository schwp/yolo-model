#pragma once 
#include "yolo/Layer.hpp"
#include <string>
#include <vector>

class FullyConnectedLayer : public Layer {
public:
    FullyConnectedLayer(int in_features, int out_features);
    std::vector<float> forward(const std::vector<float>& input, Shape in, Shape& out) override;

    void setAlpha(float alpha) { alpha_ = alpha; }
    
    // These methods are for testing purposes
    void setWeights(const std::vector<float>& weights) { weights_ = weights; }
    void setBiases(const std::vector<float>& biases) { biases_ = biases; }

private:
    int in_features_, out_features_;
    float alpha_ = 0.1f;
    std::vector<float> weights_, biases_;

    inline float leakyReLU(float x) const { return x < 0 ? alpha_ * x : x; }
};