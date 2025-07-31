#include "yolo/PoolLayer.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

static std::vector<float> maxPooling(const std::vector<float>& input, Shape in, int poolSize, int stride) {
    int outH = (in.H - poolSize) / stride + 1;
    int outW = (in.W - poolSize) / stride + 1;

    std::vector<float> output(outH * outW * in.C, 0.0f);

    for (int c = 0; c < in.C; c++) {
        for (int h = 0; h < outH; h++) {
            for (int w = 0; w < outW; w++) {
                float max = -std::numeric_limits<float>::infinity();
                for (int i = 0; i < poolSize; i++) {
                    for (int j = 0; j < poolSize; j++) {
                        int inputH = h * stride + i;
                        int inputW = w * stride + j;
                        int idx = c * in.H * in.W + inputH * in.W + inputW;
                        max = std::max(max, input[idx]);
                    }
                }
                output[c * outH * outW + h * outW + w] = max;
            }
        }
    }

    return output;
}

static std::vector<float> averagePooling(const std::vector<float>& input, Shape in, int poolSize, int stride) {
    int outH = (in.H - poolSize) / stride + 1;
    int outW = (in.W - poolSize) / stride + 1;

    std::vector<float> output(outH * outW * in.C, 0.0f);

    for (int c = 0; c < in.C; c++) {
        for (int h = 0; h < outH; h++) {
            for (int w = 0; w < outW; w++) {
                float sum = 0.0f;
                for (int i = 0; i < poolSize; i++) {
                    for (int j = 0; j < poolSize; j++) {
                        int inputH = h * stride + i;
                        int inputW = w * stride + j;
                        int idx = c * in.H * in.W + inputH * in.W + inputW;
                        sum += input[idx];
                    }
                }
                output[c * outH * outW + h * outW + w] = sum / (poolSize * poolSize);
            }
        }
    }

    return output;
}

PoolLayer::PoolLayer(std::string layerTechnique, int poolSize, int stride)
    : poolSize(poolSize), stride(stride), layerTechnique(layerTechnique) {
        if (poolSize <= 0 || stride <= 0) {
            throw std::invalid_argument("Pool size and stride must be positive integers.");
        }
    }

std::vector<float> PoolLayer::forward(const std::vector<float>& input, Shape in, Shape out) {
    if (layerTechnique == "max") {
        return maxPooling(input, in, poolSize, stride);
    } else if (layerTechnique == "average") {
        return averagePooling(input, in, poolSize, stride);
    } else {
        throw std::invalid_argument("Unknown pooling technique: " + layerTechnique);
    }
}
