cmake_minimum_required(VERSION 3.21)
project(SimpleSLAM)

set(CMAKE_CXX_STANDARD 20)

add_executable(SimpleSLAM main.cpp main.cpp)

find_package(OpenCV REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(PCL CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)

target_link_libraries(SimpleSLAM PRIVATE Eigen3::Eigen)
target_link_libraries(SimpleSLAM PRIVATE ${OpenCV_LIBS})
target_link_libraries(SimpleSLAM PRIVATE ${PCL_LIBRARIES})
target_link_libraries(SimpleSLAM PRIVATE g2o::core)