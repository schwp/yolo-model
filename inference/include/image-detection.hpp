#ifndef IMAGE_DETECTION_HPP
#define IMAGE_DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <iostream>

// YOLOv4 CONST
#define IMG_SIZE 608
#define SCORE_THRESHOLD 0.1
#define NMS_THRESHOLD 0.45

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

// Fonctions principales
std::vector<std::string> get_class_names();
void load_model(cv::dnn::Net& net);
void detect(cv::Mat &image, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className);
void display_detection(cv::Mat &img);

#endif // IMAGE_DETECTION_HPP
