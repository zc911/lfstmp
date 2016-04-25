cmake_minimum_required(VERSION 2.8)
PROJECT(Matrix_App)

IF(CMAKE_BUILD_TYPE)
ELSE()
	SET(CMAKE_BUILD_TYPE "Debug")
ENDIF()

SET(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -std=c++11 -DDEBUG -g -O3 -w")
SET(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -std=c++11 -DNDEBUG -O3 -w")

SET(PLATFORM ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})

INCLUDE_DIRECTORIES(src include ../engine/src/ ../engine/include/dgcaffe)

FIND_LIBRARY(LIB_MATRIX NAMES matrix PATHS ../engine/build/src)
FIND_LIBRARY(LIB_CAFFE NAMES caffe PATHS ../engine/lib/caffe/${PLATFORM})
SET(CUDA_LIB_PATH /usr/local/cuda/lib64)
FIND_LIBRARY(LIB_CUDART NAMES cudart PATHS ${CUDA_LIB_PATH})

LIST(APPEND LIBS ${LIB_MATRIX} glog gflags protobuf leveldb grpc++ grpc jsoncpp b64 snappy lmdb jpeg png GL glut curl uuid freetype ${LIB_TBB} boost_system hdf5_hl hdf5 m opencv_core opencv_highgui opencv_imgproc opencv_calib3d ${LIB_CUDART} ${LIB_CUDNN} ${LIB_CAFFE} ${LIB_DGMEDIA} gstreamer-0.10 gobject-2.0 gmodule-2.0 gthread-2.0 xml2 glib-2.0 gstapp-0.10 z dl pthread)


ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(tools)
#ADD_SUBDIRECTORY(example)
