#include <iostream>
#include <opencv2/opencv.hpp>
#include "image-detection.hpp"

using namespace cv;

int main(int argc, char const *argv[])
{
    VideoCapture cam(1); // 0 use the back camera of my iPhone
    if (!cam.isOpened()) {
        std::cerr << "Cannot open the camera" << std::endl;
        return -1;
    }

    std::vector<std::string> class_names = get_class_names();
    dnn::Net net;
    load_model(net);

    if (net.empty()) {
        std::cerr << "Failed to load the model" << std::endl;
        return -1;
    }

    Mat frame;
    while (true) {
        cam >> frame;
        if (frame.empty()) {
            std::cerr << "Failed to capture camera frame" << std::endl;
            break;
        }

        std::vector<Detection> detections;
        detect(frame, net, detections, class_names);
        imshow("WebCam", frame);
        if (waitKey(30) >= 0) break;
    }

    cam.release();
    destroyAllWindows();
    return 0;
}
