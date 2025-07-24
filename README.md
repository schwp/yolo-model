# YOLO Model Implementation and Inference

## Requirements
- **OpenCV 4.x** with DNN module support
- **CMake 3.10+**
- **C++ Compiler**

## Implementation
In process ...

## Inference

This project implements object detection using YOLOv4 with the COCO dataset. The inference system can detect and classify 80 different object classes in static images, or live using your computer webcam.

### Features

- **YOLOv4 Model**: Uses the powerful YOLOv4 architecture for accurate object detection
- **COCO Dataset**: Trained on 80 common object classes (person, car, bicycle, etc.)
- **Inference**: Use CPU for inference

### Model Files Setup

The project requires YOLOv4 model files in the `inference/models/` directory. Some files are included, but you need to download the weights file:

To download it, use the following command :
```bash
cd inference/models/
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
```

Or download manually from: https://github.com/AlexeyAB/darknet/releases

### Building the Project

1. **Go to the inference directory:**
   ```bash
   cd inference
   ```

2. **Build the project:**
   ```bash
   cmake . && make
   ```

   This will create the `object-detection` and `live-detection` executable.

### Usage

#### Basic Object Detection

Run object detection on a single image:

```bash
./object-detection <image_path>
```
Replace `<image_path>` with the image you want to procces an object detection on.

The `test/` directory contains sample images for testing the project.

#### Live Object Detection (Webcam)

Run live object detection using your webcam:

```bash
./live-detection
```

This will open your default webcam and display real-time object detection results in a window.
