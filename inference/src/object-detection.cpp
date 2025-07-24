#include "image-detection.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <fstream>

using namespace cv;

int main(int argc, char const *argv[]) {
    if (argc != 2) {
        std::cerr << "Error: usage " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }
    
    Mat image = imread(argv[1], IMREAD_COLOR);
    
    if (image.empty()) {
        std::cerr << "Error: Could not load image '" << argv[1] << "'" << std::endl;
        return -1;
    }

    std::vector<std::string> classNames = get_class_names();
    dnn::Net network;
    load_model(network);
    
    if (network.empty()) {
        std::cerr << "Error: Failed to load YOLOv4 model!" << std::endl;
        std::cerr << "Please ensure models/yolov4.cfg and models/yolov4.weights exist." << std::endl;
        return -1;
    }
    
    std::vector<Detection> detections;
    detect(image, network, detections, classNames);

    display_detection(image);
}
