cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)
project(trackByDetection VERSION 1.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)

add_subdirectory(detector)
add_subdirectory(sort)