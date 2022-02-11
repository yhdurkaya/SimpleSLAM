//
// Created by yhd on 11.02.2022.
//

#ifndef SIMPLESLAM_FEATUREEXTRACTOR_H
#define SIMPLESLAM_FEATUREEXTRACTOR_H

#include <iostream>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

class FeatureExtractor
{
public:
    FeatureExtractor();
    void extractKeypoints(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
private:
    void computeDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
private:
    cv::Ptr<cv::ORB> orb;
    cv::Ptr<cv::GFTTDetector> gftt;
};


#endif //SIMPLESLAM_FEATUREEXTRACTOR_H
