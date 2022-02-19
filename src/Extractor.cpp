//
// Created by yhd on 11.02.2022.
//

#include "../inc/Extractor.h"

Extractor::Extractor()
{
    orb = cv::ORB::create();
    gftt = cv::GFTTDetector::create(3000, 0.01, 5, 3);
    bruteForceMatcher = cv::BFMatcher::create(cv::NORM_HAMMING, false);
}

void Extractor::extractKeypoints(const cv::Mat& image)
{
    matches.clear();
    gftt->detect(image, currentFrameKeyPoints);

    computeDescriptors(image);

    if(previousFrameKeyPoints.empty())
    {
        saveNewFeatures();
        return;
    }
    matcher();
    filterCorrespondences();
    saveNewFeatures();
}

void Extractor::computeDescriptors(const cv::Mat& image)
{
    orb->compute(image, currentFrameKeyPoints, currentFrameDescriptors);
}

void Extractor::matcher()
{
    std::vector<std::vector<cv::DMatch>> matchesBeforeDistance;
    bruteForceMatcher->knnMatch(currentFrameDescriptors, previousFrameDescriptors, matchesBeforeDistance, 2);
    for(auto match : matchesBeforeDistance)
    {
        if(match[0].distance < 0.75*match[1].distance)
        {
            matches.push_back(std::pair(currentFrameKeyPoints[match[0].queryIdx], previousFrameKeyPoints[match[0].trainIdx]));
        }
    }
}

void Extractor::filterCorrespondences()
{
    std::vector<cv::Point2f> pointsPrevious, pointsCurrent;

    for(auto&& point : matches)
    {
        pointsCurrent.push_back(point.first.pt);
    }
    for(auto&& point : matches)
    {
        pointsPrevious.push_back(point.second.pt);
    }

    fundamentalMatrix = cv::findFundamentalMat(pointsPrevious, pointsCurrent, outputMask);
}

void Extractor::saveNewFeatures()
{
    previousFrameKeyPoints = currentFrameKeyPoints;
    previousFrameDescriptors = currentFrameDescriptors;
}

void Extractor::drawCorrespondingLines(cv::Mat &image, bool enableFalseCorrespondences)
{
    for(std::size_t i = 0; i < this->matches.size(); ++i){
        //std::cout << "Matches: " << detector->matches.size() << "\n";
        auto kp = matches[i].first;
        cv::circle(image, kp.pt, 2, cv::Scalar(0, 255, 0), 1);
        if(outputMask.at<uchar>(0, i) == 1){
            cv::line(image, matches[i].first.pt, matches[i].second.pt,
                     cv::Scalar(255, 0, 0));
        }
        if(enableFalseCorrespondences && outputMask.at<uchar>(0, i) == 0) {
            cv::line(image, matches[i].first.pt, matches[i].second.pt,
                     cv::Scalar(0, 0, 255));
        }
    }
}