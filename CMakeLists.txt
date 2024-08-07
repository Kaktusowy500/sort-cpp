cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
project(detections_tracker VERSION 1.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)

add_subdirectory(detector)
add_subdirectory(sort)


find_package(OpenCV REQUIRED)
add_executable(detections_tracker detections_tracker_example.cpp DetectionsTracker.cpp)
target_link_libraries(detections_tracker PRIVATE detector sort ${OpenCV_LIBS})
target_include_directories(detections_tracker PRIVATE detector sort ${OpenCV_INCLUDE_DIRS})
