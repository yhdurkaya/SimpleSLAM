//
// Created by yhd on 8.02.2022.
//

#include <iostream>

#include "opencv2/opencv.hpp"

#include "inc/Extractor.h"

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
        detector->drawCorrespondingLines(currentFrame, false);

        cv::imshow("SimpleSlam", currentFrame);

        char c=(char)cv::waitKey(25);

        if(c==27) {
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
}
