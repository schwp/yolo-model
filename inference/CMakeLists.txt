cmake_minimum_required(VERSION 3.10)
project(inference)

find_package(OpenCV REQUIRED)

include_directories(include)

# Executable for object detection using fixed image
add_executable(object-detection
    src/object-detection.cpp
    src/image-detection.cpp
)
target_include_directories(object-detection
    PRIVATE ${OpenCV_INCLUDE_DIRS}
    PRIVATE include
)
target_link_libraries(object-detection ${OpenCV_LIBS})
