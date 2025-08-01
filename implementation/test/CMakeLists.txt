find_package(GTest REQUIRED)

set(TEST_SOURCES
    test_pooling.cpp
    test_convolution.cpp
)

add_executable(yolo_tests ${TEST_SOURCES})

target_link_libraries(yolo_tests 
    PRIVATE
        yolo::yolo
        GTest::gtest
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(yolo_tests)
