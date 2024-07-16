#include "SortTracker.h"

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <model_path> <video_path> <device_name>" << std::endl;
        return -1;
    }

    std::string modelPath = argv[1];
    std::string videoPath = argv[2];
    std::string deviceName = argv[3];
    ObjectDetector detector(modelPath, deviceName);
    SortTracker detectionsTracker;
    int totalFrames = 0;
    double totalTime = 0.0;

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Error opening video file: " << videoPath << std::endl;
        return 1;
    }

    cv::Mat frame;

    std::vector<LabeledBox> detections;
    while (cap.read(frame))
    {
        totalFrames++;
        int64 startTime = getTickCount();
        bool detection_update = (totalFrames % 2 == 0);
        if (detection_update)
        {
            detections = detector.detect(frame);
            detectionsTracker.update(frame, detections);
        }
        auto trackingResults = detectionsTracker.getTrackingResults(!detection_update);
        detectionsTracker.drawTrackingResults(frame, trackingResults);
        cv::imshow("Tracking", frame);
        double cycleTime = (double)(getTickCount() - startTime);
        totalTime += cycleTime / getTickFrequency();
        if (cv::waitKey(50) >= 0)
        {
            break;
        }
    }

    std::cout << "Total Tracking took: " << totalTime << " for " << totalFrames << " frames or "
              << ((double)totalFrames / (double)totalTime) << " FPS" << std::endl;
    std::cout << "Last ID: " << KalmanTracker::kf_count << std::endl;

    return 0;
}