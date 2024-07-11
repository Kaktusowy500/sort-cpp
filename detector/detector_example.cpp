#include <opencv2/opencv.hpp>
#include "ObjectDetector.h"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <video_path> <device_name>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string video_path = argv[2];
    std::string device_name = argv[3];

    ObjectDetector detector(model_path, device_name);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file: " << video_path << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<LabeledBox> detections = detector.detect(frame);
        detector.draw_detections(frame, detections);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> detection_time = end - start;
        std::cout << "Detection time: " << detection_time.count() << " seconds" << std::endl;
        cv::imshow("Detections", frame);

        if (cv::waitKey(1) >= 0) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
