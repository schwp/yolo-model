#pragma once
#include <vector>

struct Shape { int C, H, W; } ;

class Layer{
public:
    virtual std::vector<float> forward(const std::vector<float>& input, Shape in, Shape out) = 0;
    virtual ~Layer() = default;
};
