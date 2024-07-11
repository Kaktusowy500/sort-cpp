cmake_minimum_required(VERSION 3.10)

project(sort VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Define source files
set(SOURCES
    main.cpp
    Hungarian.cpp
    KalmanTracker.cpp
)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
