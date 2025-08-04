#include "yolo/YoloV1Model.hpp"
#include "yolo/ConvLayer.hpp"
#include "yolo/FullyConnectedLayer.hpp"
#include "yolo/PoolLayer.hpp"
#include <iostream>

YoloV1Model::YoloV1Model(int input_channels, int S, int B, int C)
    : S_(S), B_(B), C_(C) {
    int IC = input_channels;
    layers_.reserve(29);

    layers_.emplace_back(new ConvLayer(IC, 64, 7, 2, 0));  IC = 64;
    layers_.emplace_back(new PoolLayer("max", 2, 2));

    layers_.emplace_back(new ConvLayer(IC, 192, 3, 1, 1)); IC = 192;
    layers_.emplace_back(new PoolLayer("max", 2, 2));

    layers_.emplace_back(new ConvLayer(IC, 128, 1, 1, 0)); IC = 128;
    layers_.emplace_back(new ConvLayer(IC, 256, 3, 1, 1)); IC = 256;
    layers_.emplace_back(new ConvLayer(IC, 256, 1, 1, 0)); IC = 256;
    layers_.emplace_back(new ConvLayer(IC, 512, 3, 1, 1)); IC = 512;
    layers_.emplace_back(new PoolLayer("max", 2, 2));

    for (int i = 0; i < 4; ++i) {
        layers_.emplace_back(new ConvLayer(IC, 256, 1, 1, 0)); IC = 256;
        layers_.emplace_back(new ConvLayer(IC, 512, 3, 1, 1)); IC = 512;
    }

    layers_.emplace_back(new ConvLayer(IC, 512, 1, 1, 0)); IC = 512;
    layers_.emplace_back(new ConvLayer(IC, 1024, 3, 1, 1)); IC = 1024;
    layers_.emplace_back(new PoolLayer("max", 2, 2));

    layers_.emplace_back(new ConvLayer(IC, 256, 1, 1, 0)); IC = 256;
    layers_.emplace_back(new ConvLayer(IC, 512, 3, 1, 1)); IC = 512;
    layers_.emplace_back(new ConvLayer(IC, 512, 1, 1, 0)); IC = 512;
    layers_.emplace_back(new ConvLayer(IC, 1024, 3, 1, 1)); IC = 1024;
    layers_.emplace_back(new ConvLayer(IC, 1024, 3, 2, 1)); IC = 1024;
    layers_.emplace_back(new ConvLayer(IC, 1024, 3, 1, 1)); IC = 1024;
    layers_.emplace_back(new ConvLayer(IC, 1024, 3, 1, 1)); IC = 1024;

    layers_.emplace_back(new FullyConnectedLayer(7 * 7 * 1024, 4096));
    layers_.emplace_back(new FullyConnectedLayer(4096, S_ * S_ * (C_ * (B_ * 5 + C_))));

}

YoloV1Model::YoloV1Model(const std::string& configPath) {
    // Implementation for loading the model configuration from a file
}

std::vector<float> YoloV1Model::forward(const std::vector<float>& input) {
    Shape in = {3, 448, 448}, currShape;
    std::vector<float> output = input;

    for (const auto& layer : layers_) {
        output = layer->forward(output, in, currShape);
        in = currShape;
    }
    return output;
}