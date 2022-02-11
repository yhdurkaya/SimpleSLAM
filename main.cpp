//
// Created by yhd on 8.02.2022.
//

#include <iostream>

#include "opencv2/opencv.hpp"

#include "Extractor.h"

int main()
{
    cv::Mat currentFrame, currentFrameGray;
    auto detector = std::make_unique<Extractor>();

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

        detector->extractKeypoints(currentFrameGray);

        for(std::size_t i = 0; i < detector->matches.size(); ++i){
            auto kp = detector->matches[i].first;
            cv::circle(currentFrame, kp.pt, 2, cv::Scalar(0, 255, 0), 1);
            cv::line(currentFrame, detector->matches[i].first.pt, detector->matches[i].second.pt,
                     cv::Scalar(255, 0, 0));
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
