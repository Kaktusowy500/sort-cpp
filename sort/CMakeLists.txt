# cmake_minimum_required(VERSION 3.10)

# project(sort VERSION 1.0)

# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(sort
    STATIC
    Hungarian.cpp
    KalmanTracker.cpp
)

find_package(OpenCV REQUIRED)
target_link_libraries(sort PRIVATE ${OpenCV_LIBS})

add_executable(sort_example sort_example.cpp)

target_link_libraries(sort_example PRIVATE ${OpenCV_LIBS} sort)
