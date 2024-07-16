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
#pragma once
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


class SortTracker
{
public:
    SortTracker();
    void update(cv::Mat &frame, const std::vector<LabeledBox> & detections);
    vector<TrackingBox> getTrackingResults(bool predict = false);
    void drawTrackingResults(cv::Mat & frame, const vector<TrackingBox> &frameTrackingResult);

private:
    vector<KalmanTracker> trackers;
    vector<Rect_<float>> predictedBoxes;
    vector<vector<double>> iouMatrix;
    vector<int> assignment;
    set<int> unmatchedDetections;
    set<int> unmatchedTrajectories;
    set<int> allItems;
    set<int> matchedItems;
    vector<cv::Point> matchedPairs;
    unsigned int trkNum = 0;
    unsigned int detNum = 0;

    static const int colorNum = 20;

    int frameCount = 0;
    int maxAgeWithoutUpdate = 5;
    int minHitStreakForVerification = 2;
    int maxFramesWithoutDetect = 3;
    double iouThreshold = 0.15;
    Scalar_<int> randColor[colorNum];

    double GetIOU(Rect_<float> bb1, Rect_<float> bb2);
};
