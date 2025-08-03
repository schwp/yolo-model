#include <gtest/gtest.h>
#include "yolo/FullyConnectedLayer.hpp"

TEST(FullyConnectedLayerTest, ForwardPassWithKnownWeights) {
    FullyConnectedLayer layer(2, 2);

    layer.setWeights({1.0, 2.0, 3.0, 4.0});
    layer.setBiases({0.0, 0.0});
    layer.setAlpha(0.0f);
    
    std::vector<float> input = {1.0, 1.0};
    Shape input_shape = {1, 1, 2};
    Shape output_shape = {1, 1, 2};
    
    auto output = layer.forward(input, input_shape, output_shape);
    
    ASSERT_FLOAT_EQ(output[0], 3.0);
    ASSERT_FLOAT_EQ(output[1], 7.0);
}

TEST(FullyConnectedLayerTest, ForwardPass) {
    FullyConnectedLayer layer(2, 2);

    std::vector<float> input = {1.0, 0.0};
    Shape input_shape = {1, 1, 2};
    Shape output_shape = {1, 1, 2};
    
    auto output = layer.forward(input, input_shape, output_shape);
    
    ASSERT_EQ(output.size(), 2);
    ASSERT_EQ(output_shape.C, 1);
    ASSERT_EQ(output_shape.H, 1);
    ASSERT_EQ(output_shape.W, 2);

    ASSERT_TRUE(output[0] != 0.0f || output[1] != 0.0f);
}

TEST(FullyConnectedLayerTest, OutputSize) {
    FullyConnectedLayer layer(3, 4);
    std::vector<float> input = {1.0, 2.0, 3.0};
    Shape input_shape = {1, 1, 3};
    Shape output_shape = {1, 1, 4};
    
    auto output = layer.forward(input, input_shape, output_shape);
    ASSERT_EQ(output.size(), 4);
}