cmake_minimum_required(VERSION 2.8)
PROJECT(matrix)

IF (CMAKE_BUILD_TYPE)
ELSE ()
    SET(CMAKE_BUILD_TYPE "Debug")
ENDIF ()

SET(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -std=c++11 -DDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -g  -w")
SET(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -std=c++11 -DNDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -O3 -w")

SET(PLATFORM ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})
SET(INCLUDE_PATH ${PROJECT_SOURCE_DIR}/../include)
SET(LIBS_PATH ${PROJECT_SOURCE_DIR}/../lib)

#INCLUDE_DIRECTORIES(../utils/src src include include/dgcaffe include/mxnet include/watchdog include/dgmedia-1.8.0 /usr/local/cuda/include /usr/include/glib-2.0 /usr/lib/x86_64-linux-gnu/glib-2.0/include /usr/include/gstreamer-0.10 /usr/include/glib-2.0 /usr/lib/arm-linux-gnueabihf/glib-2.0/include /usr/include/libxml2 /usr/include/freetype2 include/dgcdnn/cdnn_predict_src/convnet_predict_alignment)
execute_process(COMMAND find ${INCLUDE_PATH}/dgcdnn -type d OUTPUT_VARIABLE INLIST)
MESSAGE(${INLIST})

INCLUDE_DIRECTORIES(${INCLUDE_PATH} src ${INCLUDE_PATH}/dgcaffe ${INCLUDE_PATH}/mxnet ${INCLUDE_PATH}/watchdog ${INCLUDE_PATH}/matrix_util ${INCLUDE_PATH}/dgmedia-1.8.0 ${INCLUDE_PATH}/dgface 
        /usr/include/glib-2.0 /usr/lib/x86_64-linux-gnu/glib-2.0/include /usr/include/gstreamer-0.10 /usr/include/glib-2.0 /usr/include/opencv /usr/include/libxml2 /usr/include/freetype2
         ${INCLUDE_PATH}/dgcdnn/cdnn/mkl64 ${INCLUDE_PATH}/dgcdnn/cdnn/common ${INCLUDE_PATH}/dgcdnn/cdnn/convnet_predict_alignment ${INCLUDE_PATH}/dgcdnn/cdnn/atlas 
         ${INCLUDE_PATH}/dgcdnn/cdnn/extract_metric_learning_feature ${INCLUDE_PATH}/dgcdnn/cdnn_caffe/extract_features ${INCLUDE_PATH}/dgcdnn/cdnn_caffe/face_detect/fcn_detector
         ${INCLUDE_PATH}/dgcdnn/cdnn_caffe/face_detect/cdnn_detector ${INCLUDE_PATH}/dgcdnn/cdnn_caffe/face_align/face_align_caffe ${INCLUDE_PATH}/dgcdnn/cdnn_caffe/face_detect ${INCLUDE_PATH}/dgcdnn/cdnn_caffe/ ${INCLUDE_PATH}/dgcdnn/cdnn_caffe/face_align)


SET(CUDA_LIB_PATH /usr/local/cuda/lib64)

IF (${PLATFORM} STREQUAL "Linux-armv7l")
    INCLUDE_DIRECTORIES(include/caffe-6.5)
    SET(CUDA_LIB_PATH "/usr/local/cuda/lib")
    FIND_LIBRARY(LIB_TBB NAMES tbb PATHS lib/tbb/${PLATFORM} NO_CMAKE_SYSTEM_PATH NO_DEFAULT_PATH NO_CMAKE_ENVIRONMENT_PATH NO_CMAKE_PATH NO_SYSTEM_ENVIRONMENT_PATH NO_CMAKE_BUILDS_PATH NO_CMAKE_SYSTEM_PATH)
ELSE (${PLATFORM} STREQUAL "Linux-armv7l")
    INCLUDE_DIRECTORIES(include/caffe-7.0)
    SET(LIB_TBB tbb)
ENDIF (${PLATFORM} STREQUAL "Linux-armv7l")

OPTION(ENABLE_CUDA "" ON)
FIND_PACKAGE(CUDA REQUIRED)
FIND_LIBRARY(LIB_CAFFE NAMES caffe PATHS ${LIBS_PATH}/dgcaffe/${PLATFORM}/${CUDA_VERSION})
FIND_LIBRARY(LIB_MXNET NAMES mxnet PATHS ${LIBS_PATH}/mxnet/${PLATFORM}/${CUDA_VERSION})
FIND_LIBRARY(LIB_DLIB NAMES dlib PATHS ${LIBS_PATH}/dlib/${PLATFORM})
FIND_LIBRARY(LIB_CUDART NAMES cudart PATHS ${CUDA_LIB_PATH})
FIND_LIBRARY(LIB_THPLATEID NAMES thplateid PATHS ${LIBS_PATH}/thplateid/${PLATFORM})
FIND_LIBRARY(LIB_HASP NAMES hasp PATHS ${LIBS_PATH}/watchdog/${PLATFORM})
FIND_LIBRARY(LIB_WATCHDOG NAMES watchdog hasp PATHS ${LIBS_PATH}/watchdog/${PLATFORM})
FIND_LIBRARY(LIB_DGMEDIA NAMES dgmedia PATHS ${LIBS_PATH}/dgmedia-1.8.0/${PLATFORM})
FIND_LIBRARY(LIB_PROTOBUF NAMES protobuf PATHS ${LIBS_PATH}/protobuf/${PLATFORM})
FIND_LIBRARY(LIB_GTEST_MAIN NAMES gtest_main PATHS ${LIBS_PATH}/gtest/${PLATFORM})
FIND_LIBRARY(LIB_GTEST NAMES gtest PATHS ${LIBS_PATH}/gtest/${PLATFORM})

FIND_LIBRARY(LIB_CDNN_SCORE NAMES cdnn-score PATHS ${LIBS_PATH}/dgface/${PLATFORM}/${CUDA_VERSION}/cdnn)
FIND_LIBRARY(LIB_CDNN_FEATURE NAMES cdnn-feature PATHS ${LIBS_PATH}/dgface/${PLATFORM}/${CUDA_VERSION}/cdnn)
FIND_LIBRARY(LIB_CDNN_ALIGN NAMES cdnn-align PATHS ${LIBS_PATH}/dgface/${PLATFORM}/${CUDA_VERSION}/cdnn)

#FIND_LIBRARY(LIB_CDNN_CAFFE_FACE_ALIGNMENT NAMES facealignment PATHS ${LIBS_PATH}/dgface/${PLATFORM}/${CUDA_VERSION}/cdnn-caffe)
#FIND_LIBRARY(LIB_CDNN_CAFFE_FACE_DETECTION NAMES facedetection PATHS ${LIBS_PATH}/dgface/${PLATFORM}/${CUDA_VERSION}/cdnn-caffe)
#FIND_LIBRARY(LIB_CDNN_CAFFE_FACE_RECOGNITION NAMES facerecognition PATHS ${LIBS_PATH}/dgface/${PLATFORM}/${CUDA_VERSION}/cdnn-caffe)
FIND_LIBRARY(LIB_CDNN_CAFFE_FACE_SDK NAMES facesdk PATHS ${LIBS_PATH}/dgface/${PLATFORM}/${CUDA_VERSION}/cdnn-caffe)

FIND_LIBRARY(LIB_DGFACE NAMES dgface cdnn-score PATHS ${LIBS_PATH}/dgface/${PLATFORM}/${CUDA_VERSION})
FIND_LIBRARY(LIB_MATRIX_UTILS NAMES matrix_util PATHS ${LIBS_PATH}/matrix_util/${PLATFORM})

LIST(APPEND LIBS  ${LIB_DGFACE} ${LIB_CDNN_FEATURE} ${LIB_CDNN_ALIGN}    ${LIB_CDNN_SCORE} ${LIB_TBB} ${LIB_CDNN_CAFFE_FACE_SDK} ${LIB_DLIB} ${LIB_PROTOBUF} ${LIB_MATRIX_UTILS} ${LIB_GTEST_MAIN} ${LIB_GTEST} ${LIB_CUDART} ${LIB_MXNET} ${LIB_CUDNN} ${LIB_CAFFE} ${LIB_DGMEDIA} ${LIB_THPLATEID} ${LIB_WATCHDOG} ${LIB_HASP})
LIST(APPEND LIBS pthread glog gflags leveldb jsoncpp crypto b64 snappy lmdb jpeg png GL glut curl uuid freetype boost_filesystem boost_system hdf5_hl hdf5 m opencv_core opencv_highgui opencv_imgproc opencv_calib3d  opencv_features2d gstreamer-0.10 gobject-2.0 gmodule-2.0 gthread-2.0 xml2 glib-2.0 gstapp-0.10 z dl openblas)



ADD_SUBDIRECTORY(src)
#ADD_SUBDIRECTORY(test)
