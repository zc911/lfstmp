cmake_minimum_required(VERSION 2.8)
set(PROJECT_NAME_STR matrix)
PROJECT(${PROJECT_NAME_STR})
set(CMAKE_VERBOSE_MAKEFILE 1)

IF (APPLE)
    set(CMAKE_MACOSX_RPATH 1)
ENDIF()

IF(CMAKE_BUILD_TYPE)
ELSE()
    SET(CMAKE_BUILD_TYPE "Debug")
ENDIF()

SET(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -std=c++11 -DUSE_CUDA -DDEBUG -g -O3 -w")
SET(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -std=c++11 -DNDEBUG -O3 -w")

find_package(Threads REQUIRED)
#if(CMAKE_COMPILER_IS_GNUCXX)
#  add_definitions(-Wall -Wno-deprecated -pthread)
#endif()
message(STATUS "Found Threads libs: ${THREADS_LIBS}")

find_package(Freetype REQUIRED)
INCLUDE_DIRECTORIES(${FREETYPE_INCLUDE_DIRS})
message(STATUS "Found FreeType libs: ${FREETYPE_LIBRARIES}")

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/apps/include/pbjson ${PROJECT_SOURCE_DIR}/apps/include/Simple-Web-Server)

IF(APPLE)
    OPTION(ENABLE_CUDA "disable cuda by default" OFF)
    find_package(Caffe REQUIRED)
    INCLUDE_DIRECTORIES(${CAFFE_INCLUDE_DIRS})
    message(STATUS "Found Caffe libs: ${Caffe_LIBRARIES}")
    #
    set(OpenBLAS_DIR /usr/local/share/cmake/Modules/)
    find_package(OpenBLAS REQUIRED)
    INCLUDE_DIRECTORIES(${OpenBLAS_INCLUDE_DIR})
    message(STATUS "Found OpenBLAS libs: ${OpenBLAS_LIB}")
    set(Caffe_DEP_LIBRARIES ${Caffe_LIBRARIES})

ELSE(APPLE)
    OPTION(ENABLE_CUDA "enable cuda by default" ON)
    find_package(CUDA REQUIRED)

    FIND_LIBRARY(LIB_CAFFE NAMES caffe PATHS ${PROJECT_SOURCE_DIR}/engine/lib/caffe/Linux-x86_64)
    FIND_LIBRARY(LIB_CUDART NAMES cudart PATHS /usr/local/cuda/lib64)
    FIND_LIBRARY(LIB_DGMEDIA NAMES dgmedia PATHS ${PROJECT_SOURCE_DIR}/engine/lib/dgmedia-1.4.0/Linux-x86_64)
    FIND_LIBRARY(LIB_CBLAS NAMES cblas)
    FIND_LIBRARY(LIB_LAPACK NAMES lapack)

    INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/engine/include/dgmedia-1.4.0 /usr/include/gstreamer-0.10 /usr/include/glib-2.0 /usr/lib/x86_64-linux-gnu/glib-2.0/include /usr/lib/arm-linux-gnueabihf/glib-2.0/include /usr/include/libxml2)
    SET(Caffe_DEP_LIBRARIES ${LIB_CAFFE} ${LIB_CUDART} ${LIB_DGMEDIA} ${LIB_CBLAS} ${LIB_LAPACK})
    message(STATUS "caffe dep libraries: ${Caffe_DEP_LIBRARIES}")
ENDIF(APPLE)

find_library(LIB_PBJSON pbjson PATHS ${PROJECT_SOURCE_DIR}/apps/lib/pbjson)

find_library(PROTOBUF_LIBRARY protobuf PATHS /usr/local/lib)
#find_package(Protobuf REQUIRED)
#INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIRS})
message(STATUS "Protobuf libs: ${PROTOBUF_LIBRARY}")

#find_library(DLIB_LIBRARY NAMES dlib)
#message(STATUS "Found dlib libs: ${DLIB_LIBRARY}")

find_library(GRPC_LIBRARY NAMES grpc)
find_library(GRPCPP_LIBRARY NAMES grpc++) 
find_library(GPR_LIBRARY NAMES gpr)
find_library(GFLAGS_LIBRARY NAMES gflags)
set(GRPC_LIBRARIES ${GRPCPP_LIBRARY} ${GRPC_LIBRARY} ${GPR_LIBRARY} ${GFLAGS_LIBRARY})
message(STATUS "Grpc libs: ${GRPC_LIBRARIES}")

find_library(JSONCPP_LIBRARIES NAMES jsoncpp HINTS /usr/lib /usr/local/lib)
message(STATUS "JsonCpp libs: ${JSONCPP_LIBRARIES}")

find_library(GLOG_LIBRARIES NAMES glog)
message(STATUS "Glog libs: ${GLOG_LIBRARIES}")

find_package(OpenCV REQUIRED)
message(STATUS "opencv libs: ${OpenCV_LIBS}")

find_package(Boost COMPONENTS regex system filesystem coroutine context REQUIRED)
message(STATUS "boost library: ${Boost_LIBRARIES}")

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/apps/src)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/engine/src)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/engine/include/dgmedia-1.4.0)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/engine/include/dlib-18.18.0)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/utils/src)
#INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/ranker/src)

ADD_SUBDIRECTORY(utils)
ADD_SUBDIRECTORY(engine)
#ADD_SUBDIRECTORY(wrapper)
ADD_SUBDIRECTORY(apps)
