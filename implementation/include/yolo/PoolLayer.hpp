#pragma once
#include "Layer.hpp"
#include <string>
#include <vector>

class PoolLayer: public Layer {
public:
    PoolLayer(std::string layerTechnique,int poolSize, int stride);
    std::vector<float> forward(const std::vector<float>& input, Shape in, Shape out) override;

private:
    int poolSize, stride;
    std::string layerTechnique;
};
