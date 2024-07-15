///////////////////////////////////////////////////////////////////////////////
//  SORT: A Simple, Online and Realtime Tracker
//
//  This is a C++ reimplementation of the open source tracker in
//  https://github.com/abewley/sort
//  Based on the work of Alex Bewley, alex@dynamicdetection.com, 2016
//
//  Cong Ma, mcximing@sina.cn, 2016
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.
///////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <fstream>
#include <iomanip>
#include <unistd.h>
#include <set>
#include <chrono>

#include "Hungarian.h"
#include "KalmanTracker.h"
#include "ObjectDetector.h"

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

typedef struct TrackingBox
{
    int frame;
    int id;
    Rect_<float> box;
} TrackingBox;

double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return (double)(in / un);
}

#define CNUM 20
int totalFrames = 0;
double totalTime = 0.0;

void TestSORT(ObjectDetector &detector, const std::string &videoPath, bool display);

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <model_path> <video_path> <device_name>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string video_path = argv[2];
    std::string device_name = argv[3];

    ObjectDetector detector(model_path, device_name);

    TestSORT(detector, video_path, true);

    std::cout << "Total Tracking took: " << totalTime << " for " << totalFrames << " frames or "
              << ((double)totalFrames / (double)totalTime) << " FPS" << std::endl;
    std::cout << "Last ID: " << KalmanTracker::kf_count << std::endl;

    return 0;
}

void TestSORT(ObjectDetector &detector, const std::string &videoPath, bool display)
{
    std::cout << "Processing video: " << videoPath << "..." << std::endl;

    RNG rng(0xFFFFFFFF);
    Scalar_<int> randColor[CNUM];
    for (int i = 0; i < CNUM; i++)
        rng.fill(randColor[i], RNG::UNIFORM, 0, 256);

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Error opening video file: " << videoPath << std::endl;
        return;
    }

    vector<KalmanTracker> trackers;
    KalmanTracker::kf_count = 0;

    vector<Rect_<float>> predictedBoxes;
    vector<vector<double>> iouMatrix;
    vector<int> assignment;
    set<int> unmatchedDetections;
    set<int> unmatchedTrajectories;
    set<int> allItems;
    set<int> matchedItems;
    vector<cv::Point> matchedPairs;
    vector<TrackingBox> frameTrackingResult;
    unsigned int trkNum = 0;
    unsigned int detNum = 0;

    double cycleTime = 0.0;
    int64 startTime = 0;

    cv::Mat frame;
    int frameCount = 0;
    int maxAgeWithoutUpdate = 5;
    int minHitStreakForVerification = 2;
    int maxFramesWithoutDetect = 3;
    double iouThreshold = 0.15;
    std::vector<LabeledBox> detections;

    while (cap.read(frame))
    {
        totalFrames++;
        frameCount++;
        startTime = getTickCount();

        // if (frameCount % 2 == 0) // Update detector every n frames , TODO might confuse the kalman filter, doesnt make sense to update state
        detections = detector.detect(frame);

        if (trackers.size() == 0)
        {
            // initialize kalman trackers using first detections.
            for (const auto &det : detections)
            {
                KalmanTracker trk = KalmanTracker(det.rect);
                trackers.push_back(trk);
            }
            continue;
        }

        ///////////////////////////////////////
        // 3.1. get predicted locations from existing trackers.
        predictedBoxes.clear();

        for (auto it = trackers.begin(); it != trackers.end();)
        {
            Rect_<float> pBox = it->predict();
            if (pBox.x >= 0 && pBox.y >= 0)
            {
                predictedBoxes.push_back(pBox);
                it++;
            }
            else
            {
                it = trackers.erase(it);
            }
        }

        ///////////////////////////////////////
        // 3.2. associate detections to tracked object (both represented as bounding boxes)
        // dets : detFrameData[fi]
        trkNum = predictedBoxes.size();
        detNum = detections.size();

        iouMatrix.clear();
        iouMatrix.resize(trkNum, vector<double>(detNum, 0));

        for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
        {
            for (unsigned int j = 0; j < detNum; j++)
            {
                // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
                iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detections[j].rect);
            }
        }

        // solve the assignment problem using hungarian algorithm.
        // the resulting assignment is [track(prediction) : detection], with len=preNum
        HungarianAlgorithm hungAlgo;
        assignment.clear();
        hungAlgo.Solve(iouMatrix, assignment);

        // find matches, unmatched_detections and unmatched_predictions
        unmatchedTrajectories.clear();
        unmatchedDetections.clear();
        allItems.clear();
        matchedItems.clear();

        if (detNum > trkNum) //	there are unmatched detections
        {
            for (unsigned int n = 0; n < detNum; n++)
                allItems.insert(n);

            for (unsigned int i = 0; i < trkNum; ++i)
                matchedItems.insert(assignment[i]);

            set_difference(allItems.begin(), allItems.end(),
                           matchedItems.begin(), matchedItems.end(),
                           insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
        }
        else if (detNum < trkNum) // there are unmatched trajectory/predictions
        {
            for (unsigned int i = 0; i < trkNum; ++i)
                if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                    unmatchedTrajectories.insert(i);
        }

        // filter out matched with low IOU
        matchedPairs.clear();
        for (unsigned int i = 0; i < trkNum; ++i)
        {
            if (assignment[i] == -1)
                continue;
            if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
            {
                unmatchedTrajectories.insert(i);
                unmatchedDetections.insert(assignment[i]);
            }
            else
            {
                matchedPairs.push_back(cv::Point(i, assignment[i]));
            }
        }
        ///////////////////////////////////////
        // 3.3. updating trackers

        // update matched trackers with assigned detections.
        // each prediction is corresponding to a tracker
        for (unsigned int i = 0; i < matchedPairs.size(); i++)
        {
            int trkIdx = matchedPairs[i].x;
            int detIdx = matchedPairs[i].y;
            trackers[trkIdx].update(detections[detIdx].rect);
        }

        for (auto umd : unmatchedDetections)
        {
            KalmanTracker tracker = KalmanTracker(detections[umd].rect);
            trackers.push_back(tracker);
        }

        // get trackers' output
        frameTrackingResult.clear();
        for (auto it = trackers.begin(); it != trackers.end();)
        {
            if(!it->m_verified && it->m_hit_streak >= minHitStreakForVerification)
            {
                it->m_verified = true;
            }
            if(it->m_id == 0)
                std::cout << "time since update: " << it->m_time_since_update << " hits " << it->m_hits << " hit streak " << it->m_hit_streak << " age " << it->m_age << std::endl;

            // remove dead tracklet
            if (it != trackers.end() && it->m_time_since_update > maxAgeWithoutUpdate)
            {
                // std::cout << "ID "<< it->m_id << " time since update: " << it->m_time_since_update << " hits " << it->m_hits << " hit streak " << it->m_hit_streak << " age " << it->m_age << std::endl;
                it = trackers.erase(it);
            }    
            else if ((it->m_time_since_update <= maxFramesWithoutDetect) &&
                (it->m_verified || frameCount <= minHitStreakForVerification))
            {
                TrackingBox res;
                res.box = it->get_state();
                res.id = it->m_id;
                res.frame = frameCount;
                frameTrackingResult.push_back(res);
                it++;
            }
            else
            {
                it++;
            }

        }

        cycleTime = (double)(getTickCount() - startTime);
        totalTime += cycleTime / getTickFrequency();

        if (display)
        {
            for (const auto &tb : frameTrackingResult)
            {
                cv::rectangle(frame, tb.box, randColor[tb.id % CNUM], 2, 8, 0);
                cv::putText(frame, std::to_string(tb.id), cv::Point(tb.box.x, tb.box.y - 10), FONT_HERSHEY_SIMPLEX, 0.75, randColor[tb.id % CNUM], 2);
            }
            // detector.draw_detections(frame, detections);
            cv::imshow("Tracking", frame);

            if (cv::waitKey(50) >= 0)
            {
                break;
            }
        }
    }

    if (display)
    {
        cv::destroyAllWindows();
    }
}
