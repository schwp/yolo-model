cmake_minimum_required(VERSION 3.10)
project(implementation)

set(CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build tests" ON)

set(YOLO_SOURCES
    src/PoolLayer.cpp
)

set(YOLO_HEADERS
    include/yolo/yolo.hpp
    include/yolo/Layer.hpp
    include/yolo/PoolLayer.hpp
)

add_library(yolo STATIC ${YOLO_SOURCES} ${YOLO_HEADERS})
add_library(yolo::yolo ALIAS yolo)

target_include_directories(yolo
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        src
)

target_compile_features(yolo PUBLIC cxx_std_17)
target_compile_options(yolo PRIVATE -Wall -Wextra -Wpedantic)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
