add_library(detector
	STATIC
	ObjectDetector.cpp
)
find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
target_link_libraries(detector PRIVATE openvino::runtime ${OpenCV_LIBS})

add_executable(detector_example detector_example.cpp)
target_link_libraries(detector_example PRIVATE detector ${OpenCV_LIBS})
