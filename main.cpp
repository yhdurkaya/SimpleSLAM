//
// Created by yhd on 8.02.2022.
//

#include <iostream>

#include "opencv2/opencv.hpp"

#include "FeatureExtractor.h"

int main()
{
    cv::Mat currentFrame, currentFrameGray;
    auto orbDetector = std::make_unique<FeatureExtractor>();

    cv::VideoCapture cap("../TestData/test_countryroad.mp4");

    if(!cap.isOpened())
    {
        std::cout << "Wrong input!\n";
    }

    while(true)
    {
        cap >> currentFrame;

        if(currentFrame.empty())
        {
            break;
        }

        cv::resize(currentFrame, currentFrame, cv::Size(1920/2, 1080/2));
        cv::cvtColor(currentFrame, currentFrameGray, cv::COLOR_BGR2GRAY);
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        orbDetector->extractKeypoints(currentFrameGray, keypoints, descriptors);
        std::cout << "Keypoints size: " << keypoints.size() << "\n";

        for(auto&& kp: keypoints){
            cv::circle(currentFrame, kp.pt, 2, cv::Scalar(0, 255, 0), 1);
        }

        cv::imshow("SimpleSlam", currentFrame);

        char c=(char)cv::waitKey(25);

        if(c==27) {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
}
