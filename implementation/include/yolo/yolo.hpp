#pragma once

#include <string>
#include <vector>
#include "Layer.hpp"

namespace yolov1 {
struct Detection { int class_id; float conf, x, y, w, h; };

class YOLO
{
private:
    std::vector<Layer*> layers;
    void buildNetwork();
public:
    YOLO(const std::string& configPath);
    std::vector<Detection> detect(const std::string& img, float confThr, float nmsThr);
};
} // namespace yolov1
