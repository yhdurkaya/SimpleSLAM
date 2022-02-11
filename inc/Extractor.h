//
// Created by yhd on 11.02.2022.
//

#ifndef SIMPLESLAM_EXTRACTOR_H
#define SIMPLESLAM_EXTRACTOR_H

#include <iostream>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

class Extractor
{
public:
    Extractor();
    void extractKeypoints(const cv::Mat& image);
public:
    cv::Mat currentFrameDescriptors, previousFrameDescriptors;
    std::vector<cv::KeyPoint> currentFrameKeyPoints, previousFrameKeyPoints;
    std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> matches;
private:
    cv::Ptr<cv::ORB> orb;
    cv::Ptr<cv::GFTTDetector> gftt;
    cv::Ptr<cv::DescriptorMatcher> bruteForceMatcher;
private:
    void computeDescriptors(const cv::Mat& image);
    void matcher();
    void saveNewFeatures();
};


#endif //SIMPLESLAM_EXTRACTOR_H
