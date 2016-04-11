cmake_minimum_required(VERSION 2.8)
PROJECT(matrix)

IF(CMAKE_BUILD_TYPE)
ELSE()
	SET(CMAKE_BUILD_TYPE "Debug")
ENDIF()

SET(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -std=c++11 -DDEBUG -g -O3 -w")
SET(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -std=c++11 -DNDEBUG -O3 -w")

SET(PLATFORM ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})


INCLUDE_DIRECTORIES(../utils/src src include include/dgmedia-1.4.0 /usr/local/cuda/include /usr/include/glib-2.0 /usr/lib/x86_64-linux-gnu/glib-2.0/include /usr/include/gstreamer-0.10 /usr/include/glib-2.0 /usr/lib/arm-linux-gnueabihf/glib-2.0/include /usr/include/libxml2 /usr/include/freetype2)
SET(CUDA_LIB_PATH /usr/local/cuda/lib64)

IF(${PLATFORM} STREQUAL "Linux-armv7l")
	INCLUDE_DIRECTORIES(include/caffe-6.5)
	SET(CUDA_LIB_PATH "/usr/local/cuda/lib")
	FIND_LIBRARY(LIB_TBB NAMES tbb PATHS lib/tbb/${PLATFORM} NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_PATH NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_BUILDS_PATH NO_CMAKE_SYSTEM_PATH)
ELSE(${PLATFORM} STREQUAL "Linux-armv7l")
	INCLUDE_DIRECTORIES(include/caffe-7.0)
	SET(LIB_TBB tbb)
ENDIF(${PLATFORM} STREQUAL "Linux-armv7l")

FIND_LIBRARY(LIB_DGMEDIA NAMES dgmedia PATHS lib/dgmedia-1.4.0/${PLATFORM})
FIND_LIBRARY(LIB_CAFFE NAMES caffe PATHS lib/caffe/${PLATFORM} )
FIND_LIBRARY(LIB_CUDART NAMES cudart PATHS ${CUDA_LIB_PATH})

LIST(APPEND LIBS glog gflags protobuf leveldb grpc++ grpc jsoncpp b64 snappy lmdb jpeg png GL glut curl uuid freetype ${LIB_TBB} boost_system hdf5_hl hdf5 m opencv_core opencv_highgui opencv_imgproc opencv_calib3d ${LIB_CUDART} ${LIB_CUDNN} ${LIB_CAFFE} ${LIB_DGMEDIA} gstreamer-0.10 gobject-2.0 gmodule-2.0 gthread-2.0 xml2 glib-2.0 gstapp-0.10 z dl pthread)


ADD_SUBDIRECTORY(src)
