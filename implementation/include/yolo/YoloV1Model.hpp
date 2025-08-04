#pragma once

#include <string>
#include <vector>
#include "Layer.hpp"

struct Detection { int class_id; float conf, x, y, w, h; };

class YoloV1Model
{
private:
    int S_, B_, C_;
    std::vector<std::unique_ptr<Layer>> layers_;

public:
    YoloV1Model(int input_channels, int S, int B, int C);
    YoloV1Model(const std::string& configPath);
    std::vector<float> forward(const std::vector<float>& input);
};
