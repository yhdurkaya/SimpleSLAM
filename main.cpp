//
// Created by yhd on 8.02.2022.
//

#include <iostream>

#include "opencv2/opencv.hpp"
#include "Eigen/Dense"
#include <pcl/pcl_config.h>
#include "g2o/config.h"

int main() {
    std::cout << "Hello, World!" << std::endl;

    std::cout << "OpenCV version : " << CV_VERSION << std::endl;
    std::cout << "Major version : " << CV_MAJOR_VERSION << std::endl;
    std::cout << "Minor version : " << CV_MINOR_VERSION << std::endl;
    std::cout << "Subminor version : " << CV_SUBMINOR_VERSION << std::endl;

    std::cout << "Point Cloud Library version: " << PCL_VERSION << std::endl;

    Eigen::MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
    return 0;
}
