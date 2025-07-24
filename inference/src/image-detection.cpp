#include "image-detection.hpp"
#include <fstream>

using namespace cv;

std::vector<std::string> get_class_names() {
    std::vector<std::string> class_names;
    std::ifstream class_file("models/coco.names");
    
    std::string line;
    while (std::getline(class_file, line)) {
        if (!line.empty()) class_names.push_back(line);
    }
    
    class_file.close();

    return class_names;
}

void load_model(dnn::Net& net) {
    auto result = dnn::readNetFromDarknet("models/yolov4.cfg", "models/yolov4.weights");

    if (result.empty()) {
        std::cerr << "Error loading the model" << std::endl;
        return;
    }

    result.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
    result.setPreferableTarget(dnn::DNN_TARGET_CPU);
    
    net = result;
}

// Fonction pour calculer les coordonnées des boîtes englobantes
bool calculateBoundingBox(float x, float y, float w, float h, 
                         const Mat& input_image, float x_factor, float y_factor,
                         int& left, int& top, int& width, int& height, 
                         float confidence) {
    
    bool coords_valid = false;

    if (x >= 0.0f && x <= 1.0f && y >= 0.0f && y <= 1.0f && 
        w >= 0.0f && w <= 1.0f && h >= 0.0f && h <= 1.0f) {
        
        left = int((x - 0.5 * w) * input_image.cols);
        top = int((y - 0.5 * h) * input_image.rows);
        width = int(w * input_image.cols);
        height = int(h * input_image.rows);
        coords_valid = true;
    }

    else if (x >= 0.0f && x <= IMG_SIZE && y >= 0.0f && y <= IMG_SIZE &&
             w >= 0.0f && w <= IMG_SIZE && h >= 0.0f && h <= IMG_SIZE) {
        
        left = int((x - 0.5 * w) * x_factor);
        top = int((y - 0.5 * h) * y_factor);
        width = int(w * x_factor);
        height = int(h * y_factor);
        coords_valid = true;
    }

    else {
        left = int(x - 0.5 * w);
        top = int(y - 0.5 * h);
        width = int(w);
        height = int(h);

        if (left >= 0 && top >= 0 && width > 0 && height > 0 &&
            left + width <= input_image.cols && top + height <= input_image.rows) {
            coords_valid = true;
        }
    }

    bool final_valid = coords_valid && width > 0 && height > 0 && 
                       left >= 0 && top >= 0 && 
                       left + width <= input_image.cols && 
                       top + height <= input_image.rows;
    
    
    return final_valid;
}

// Fonction pour traiter une seule sortie du réseau
void processNetworkOutput(const Mat& output, size_t output_idx, 
                         const std::vector<std::string>& className,
                         const Mat& input_image, float x_factor, float y_factor,
                         std::vector<int>& class_ids, 
                         std::vector<float>& confidences, 
                         std::vector<Rect>& boxes) {
    
    float *output_data = (float *)output.data;
    
    int out_rows, out_dimensions;
    if (output.dims == 2) {
        out_rows = output.size[0];
        out_dimensions = output.size[1];
    } else if (output.dims == 3) {
        out_rows = output.size[1];
        out_dimensions = output.size[2];
    } else {
        return;
    }

    int max_detections = std::min(out_rows, 1000);
    for (int i = 0; i < max_detections; ++i) {
        if (i * out_dimensions + 4 >= output.total()) {
            std::cout << "Warning: early exit at index " << i << " to prevent overflow" << std::endl;
            break;
        }

        float confidence = output_data[4];

        if (confidence >= SCORE_THRESHOLD && confidence <= 1.0f) {
            int available_classes = out_dimensions - 5;
            if (available_classes <= 0) {
                output_data += out_dimensions;
                continue;
            }
            
            // Trouver la classe avec le score le plus élevé
            float *classes_scores = output_data + 5;
            int num_classes = std::min(static_cast<int>(className.size()), available_classes);

            if (num_classes <= 0) {
                output_data += out_dimensions;
                continue;
            }
            
            Mat scores(1, num_classes, CV_32FC1, classes_scores);
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            if (class_id.x < 0 || class_id.x >= className.size()) {
                output_data += out_dimensions;
                continue;
            }

            // Compute the box coordinates
            if (max_class_score > SCORE_THRESHOLD && max_class_score <= 1.0) {
                float x = output_data[0];
                float y = output_data[1];
                float w = output_data[2];
                float h = output_data[3];
                
                int left, top, width, height;
                if (calculateBoundingBox(x, y, w, h, input_image, x_factor, y_factor, 
                                       left, top, width, height, confidence)) {
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }

        output_data += out_dimensions;
    }
}

void applyNMSAndDraw(std::vector<int>& class_ids, std::vector<float>& confidences, 
                     std::vector<Rect>& boxes, const std::vector<std::string>& className,
                     Mat& image, std::vector<Detection>& output) {
    std::vector<int> nms_result;
    if (!boxes.empty() && !confidences.empty())
        dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);

    // Draw the boxes filtered by NMS
    for (size_t i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        
        if (idx < 0 || idx >= class_ids.size() || idx >= boxes.size() || idx >= confidences.size())
            continue;

        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);

        if (result.class_id >= 0 && result.class_id < className.size()) {
            rectangle(image, boxes[idx], Scalar(0, 255, 0), 2);
            std::string label = className[class_ids[idx]] + " " + 
                               std::to_string(static_cast<int>(confidences[idx] * 100)) + "%";
            putText(image, label, Point(boxes[idx].x, boxes[idx].y - 5), 
                   FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
        }
    }
}

void detect(Mat &image, dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className) {    
    if (net.empty()) {
        std::cerr << "Error: No model loaded" << std::endl;
        return;
    }
    
    Mat blob;
    dnn::blobFromImage(image, blob, 1./255., Size(IMG_SIZE, IMG_SIZE), Scalar(), true, false);
    
    net.setInput(blob);
    std::vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    if (outputs.empty()) {
        std::cerr << "Erreur: no output from the network" << std::endl;
        return;
    }

    float x_factor = static_cast<float>(image.cols) / static_cast<float>(IMG_SIZE);
    float y_factor = static_cast<float>(image.rows) / static_cast<float>(IMG_SIZE);

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    
    // Processing of each network output
    for (size_t output_idx = 0; output_idx < outputs.size(); output_idx++) {
        processNetworkOutput(outputs[output_idx], output_idx, className,
                            image, x_factor, y_factor,
                            class_ids, confidences, boxes);
    }

    // Apply NMS and draw final results
    applyNMSAndDraw(class_ids, confidences, boxes, className, image, output);
}

void display_detection(Mat &img) {
    namedWindow("Image with detections", WINDOW_AUTOSIZE);
    imshow("Image with detections", img);
    waitKey(0);
    destroyAllWindows();
}
