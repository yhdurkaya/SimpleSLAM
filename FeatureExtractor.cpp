//
// Created by yhd on 11.02.2022.
//

#include "FeatureExtractor.h"

FeatureExtractor::FeatureExtractor()
{
    orb = cv::ORB::create();
    gftt = cv::GFTTDetector::create(3000, 0.01, 5, 3);
}

void FeatureExtractor::extractKeypoints(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    gftt->detect(image, keypoints);
    computeDescriptors(image, keypoints, descriptors);
}

void FeatureExtractor::computeDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    orb->compute(image, keypoints, descriptors);
}