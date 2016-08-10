cmake_minimum_required(VERSION 2.8)
PROJECT(matrix_util)

IF(CMAKE_BUILD_TYPE)
ELSE()
    SET(CMAKE_BUILD_TYPE "Debug")
ENDIF()

SET(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -std=c++11 -DDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -g -O3 -w -fprofile-arcs -ftest-coverage")
SET(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -std=c++11 -DNDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -O3 -w")

SET(PLATFORM ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})

INCLUDE_DIRECTORIES(src /usr/include/freetype2)
LIST(APPEND LIBS glog gflags jsoncpp curl uuid freetype opencv_core opencv_highgui opencv_imgproc opencv_calib3d opencv_features2d pthread)

ADD_SUBDIRECTORY(src)
#ADD_SUBDIRECTORY(test)

