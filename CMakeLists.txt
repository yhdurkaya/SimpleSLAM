cmake_minimum_required(VERSION 3.21)
project(SimpleSLAM)

set(CMAKE_CXX_STANDARD 20)

add_executable(SimpleSLAM main.cpp main.cpp src/Extractor.cpp inc/Extractor.h)

find_package(OpenCV REQUIRED)

target_link_libraries(SimpleSLAM PRIVATE ${OpenCV_LIBS})