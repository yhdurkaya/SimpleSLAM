//
// Created by yhd on 11.02.2022.
//

#ifndef SIMPLESLAM_EXTRACTOR_H
#define SIMPLESLAM_EXTRACTOR_H

#include <iostream>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

class Extractor
{
public:
    Extractor();
    void extractKeypoints(const cv::Mat& image);
    void drawCorrespondingLines(cv::Mat& image, bool enableFalseCorrespondences);
public:
    std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> matches;
    cv::Mat outputMask;
private:
    void computeDescriptors(const cv::Mat& image);
    void matcher();
    void filterCorrespondences();
    void saveNewFeatures();
private:
    cv::Ptr<cv::ORB> orb;
    cv::Ptr<cv::GFTTDetector> gftt;
    cv::Ptr<cv::DescriptorMatcher> bruteForceMatcher;

    cv::Mat fundamentalMatrix;

    cv::Mat currentFrameDescriptors, previousFrameDescriptors;
    std::vector<cv::KeyPoint> currentFrameKeyPoints, previousFrameKeyPoints;
};


#endif //SIMPLESLAM_EXTRACTOR_H
