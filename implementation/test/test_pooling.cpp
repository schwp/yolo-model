#include <gtest/gtest.h>
#include <vector>
#include <string>

#include "yolo/Layer.hpp"
#include "yolo/PoolLayer.hpp"

TEST(MaxPoolingTest, MaxPooling1D_1) {
    PoolLayer layer("max", 2, 2);
    std::vector<float> input = { 2, 2, 7, 3,
                                9, 4, 6, 1,
                                8, 5, 2, 4,
                                3, 1, 2, 6 };
    Shape inShape = { 1, 4, 4 };
    Shape outShape = { 1, 2, 2 };
    std::vector<float> output = layer.forward(input, inShape, outShape);

    ASSERT_EQ(output.size(), 4);
    ASSERT_EQ(output, 
        std::vector<float>({ 9, 7,
                            8, 6 }));
}

TEST(MaxPoolingTest, MaxPooling1D_2) {
    PoolLayer layer("max", 2, 1);
    std::vector<float> input = { 2, 2, 7, 3,
                                9, 4, 6, 1,
                                8, 5, 2, 4,
                                3, 1, 2, 6 };
    Shape inShape = { 1, 4, 4 };
    Shape outShape = { 1, 3, 3 };
    std::vector<float> output = layer.forward(input, inShape, outShape);

    ASSERT_EQ(output.size(), 9);
    ASSERT_EQ(output,
        std::vector<float>({ 9, 7, 7,
                            9, 6, 6,
                            8, 5, 6 }));
}

TEST(MaxPoolingTest, MaxPooling2D_1) {
    PoolLayer layer("max", 2, 1);
    std::vector<float> input = { 1, 2, 3, 
                                4, 5, 6,
                                7, 8, 9, 
                                
                                10, 11, 12,
                                13, 14, 15,
                                16, 17, 18 };
    Shape inShape = { 2, 3, 3 };
    Shape outShape = { 2, 2, 2 };
    std::vector<float> output = layer.forward(input, inShape, outShape);

    ASSERT_EQ(output.size(), 8);
    ASSERT_EQ(output,
        std::vector<float>({ 5, 6,
                            8, 9,

                            14, 15,
                            17, 18 }));
}

TEST(AveragePoolingTest, AveragePooling1D_1) {
    PoolLayer layer("average", 2, 2);
    std::vector<float> input = { 2, 2, 7, 3,
                                9, 4, 6, 1,
                                8, 5, 2, 4,
                                3, 1, 2, 6 };
    Shape inShape = { 1, 4, 4 };
    Shape outShape = { 1, 2, 2 };
    std::vector<float> output = layer.forward(input, inShape, outShape);

    ASSERT_EQ(output.size(), 4);
    ASSERT_EQ(output,
        std::vector<float>({ (2 + 2 + 9 + 4) / 4.0f, (7 + 3 + 6 + 1) / 4.0f,
                            (8 + 5 + 3 + 1) / 4.0f, (2 + 4 + 2 + 6) / 4.0f }));
}

TEST(AveragePoolingTest, AveragePooling1D_2) {
    PoolLayer layer("average", 2, 1);
    std::vector<float> input = { 2, 2, 7, 3,
                                9, 4, 6, 1,
                                8, 5, 2, 4,
                                3, 1, 2, 6 };
    Shape inShape = { 1, 4, 4 };
    Shape outShape = { 1, 3, 3 };
    std::vector<float> output = layer.forward(input, inShape, outShape);

    ASSERT_EQ(output.size(), 9);
    ASSERT_EQ(output,
        std::vector<float>({ (2 + 2 + 9 + 4) / 4.0f, (2 + 7 + 4 + 6) / 4.0f, (7 + 3 + 6 + 1) / 4.0f,
                            (9 + 4 + 8 + 5) / 4.0f, (4 + 6 + 5 + 2) / 4.0f, (6 + 1 + 2 + 4) / 4.0f,
                            (8 + 5 + 3 + 1) / 4.0f, (5 + 2 + 1 + 2) / 4.0f, (2 + 4 + 2 + 6) / 4.0f }));
}
