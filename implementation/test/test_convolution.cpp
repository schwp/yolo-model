#include <gtest/gtest.h>
#include <vector>

#include "yolo/Layer.hpp"
#include "yolo/ConvLayer.hpp"

TEST(ConvLayerTest, ForwardPass1D_1) {
    ConvLayer layer(1, 1, 3, 1, 1);
    std::vector<float> input = { 1, 2, 3,
                                 4, 5, 6,
                                 7, 8, 9 };
    Shape inShape = { 1, 3, 3 };
    Shape outShape;
    std::vector<float> output = layer.forward(input, inShape, outShape);
    ASSERT_EQ(outShape.C, 1);
    ASSERT_EQ(outShape.H, 3);
    ASSERT_EQ(outShape.W, 3);
    ASSERT_EQ(output.size(), 9);
}

TEST(ConvLayerTest, ForwardPass1D_2) {
    std::vector<float> filter = { 1, 0, -1,
                                 1, 0, -1,
                                 1, 0, -1 };

    std::vector<float> input = { 3, 0, 1, 2, 7, 4,
                                 1, 5, 8, 9, 3, 1,
                                 2, 7, 2, 5, 1, 3,
                                 0, 1, 3, 1, 7, 8,
                                 4, 2, 1, 6, 2, 8,
                                 2, 4, 5, 2, 3, 9 };

    std::vector<float> expected_output = { -5, -4, 0, 8,
                                         -10, -2, 2, 3,
                                         0, -2, -4, -7,
                                         -3, -2, -3, -16 };                        

    ConvLayer layer(1, 1, 3, 1, 0);
    layer.setWeights(filter);
    layer.setAlpha(1.0f);

    Shape inShape = { 1, 6, 6 };
    Shape outShape;
    std::vector<float> output = layer.forward(input, inShape, outShape);
    ASSERT_EQ(outShape.C, 1);
    ASSERT_EQ(outShape.H, 4);
    ASSERT_EQ(outShape.W, 4);
    ASSERT_EQ(output.size(), 16);
    ASSERT_EQ(output, expected_output);
}
