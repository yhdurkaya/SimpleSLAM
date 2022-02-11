//
// Created by yhd on 11.02.2022.
//

#include "Extractor.h"

Extractor::Extractor()
{
    orb = cv::ORB::create();
    gftt = cv::GFTTDetector::create(3000, 0.01, 5, 3);
    bruteForceMatcher = cv::BFMatcher::create();
}

void Extractor::extractKeypoints(const cv::Mat& image)
{
    gftt->detect(image, currentFrameKeyPoints);

    computeDescriptors(image);

    if(previousFrameKeyPoints.empty())
    {
        saveNewFeatures();
        return;
    }
    matcher();
    saveNewFeatures();
}

void Extractor::computeDescriptors(const cv::Mat& image)
{
    orb->compute(image, currentFrameKeyPoints, currentFrameDescriptors);
}

void Extractor::matcher()
{
    bruteForceMatcher->match(currentFrameDescriptors, previousFrameDescriptors, matches);
}

void Extractor::saveNewFeatures()
{
    previousFrameKeyPoints = currentFrameKeyPoints;
    previousFrameDescriptors = currentFrameDescriptors;
}