#include "SortTracker.h"

SortTracker::SortTracker()
{
    // Generate random colors
    RNG rng(0xFFFFFFFF);
    for (int i = 0; i < colorNum; i++)
        rng.fill(randColor[i], RNG::UNIFORM, 0, 256);
}

void SortTracker::update(cv::Mat &frame, const std::vector<LabeledBox> &detections)
{
    if (trackers.size() == 0)
    {
        // initialize kalman trackers using first detections.
        for (const auto &det : detections)
        {
            KalmanTracker trk = KalmanTracker(det.rect);
            trackers.push_back(trk);
        }
        return;
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
}

vector<TrackingBox> SortTracker::getTrackingResults(bool predict)
{
    frameCount++;
    vector<TrackingBox> frameTrackingResult;
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        if (!it->m_verified && it->m_hit_streak >= minHitStreakForVerification)
        {
            it->m_verified = true;
        }
        // if (it->m_id == 0)
        //     std::cout << "time since update: " << it->m_time_since_update << " hits " << it->m_hits << " hit streak " << it->m_hit_streak << " age " << it->m_age << std::endl;

        // remove dead tracklet
        if (it != trackers.end() && it->m_time_since_update > maxAgeWithoutUpdate)
        {
            it = trackers.erase(it);
        }
        else if ((it->m_time_since_update <= maxFramesWithoutDetect) &&
                 (it->m_verified || frameCount <= minHitStreakForVerification))
        {
            TrackingBox res;
            if (predict)
                res.box = it->predict();
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
    return frameTrackingResult;
}

void SortTracker::drawTrackingResults(cv::Mat &frame, const vector<TrackingBox> &frameTrackingResult)
{
    for (const auto &tb : frameTrackingResult)
    {
        cv::rectangle(frame, tb.box, randColor[tb.id % colorNum], 2, 8, 0);
        cv::putText(frame, std::to_string(tb.id), cv::Point(tb.box.x, tb.box.y - 10), FONT_HERSHEY_SIMPLEX, 0.75, randColor[tb.id % colorNum], 2);
    }
}

double SortTracker::GetIOU(Rect_<float> bb1, Rect_<float> bb2)
{
    float in = (bb1 & bb2).area();
    float un = bb1.area() + bb2.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return (double)(in / un);
}