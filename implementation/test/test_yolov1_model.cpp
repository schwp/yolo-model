#include <gtest/gtest.h>
#include "yolo/YoloV1Model.hpp"

TEST(YoloV1ModelTest, ForwardPass) {
    YoloV1Model model(3, 7, 2, 20);
    std::vector<float> input(3 * 448 * 448);
    std::vector<float> output = model.forward(input);
    EXPECT_EQ(output.size(), 7 * 7 * (20 * (2 * 5 + 20)));
}
