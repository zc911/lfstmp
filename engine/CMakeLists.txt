cmake_minimum_required(VERSION 2.8)
PROJECT(matrix)

IF (CMAKE_BUILD_TYPE)
ELSE ()
    SET(CMAKE_BUILD_TYPE "Debug")
ENDIF ()

SET(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -std=c++11 -DDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -g -O3 -w")
SET(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -std=c++11 -DNDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -g -O3 -w")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

SET(PLATFORM ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})
SET(INCLUDE_PATH ${PROJECT_SOURCE_DIR}/../include)
SET(LIBS_PATH ${PROJECT_SOURCE_DIR}/../lib)

INCLUDE_DIRECTORIES(${CONAN_INCLUDE_DIRS} ${INCLUDE_PATH}/matrix_util src
        /usr/include/glib-2.0 /usr/lib/x86_64-linux-gnu/glib-2.0/include
        /usr/include/gstreamer-0.10 /usr/include/glib-2.0 /usr/include/libxml2
        /usr/include/freetype2)

foreach (libdir ${CONAN_LIB_DIRS})
    LIST(APPEND LIBSDIR ${libdir}/${PLATFORM})
endforeach (libdir)

LINK_DIRECTORIES(${LIBSDIR} ${CONAN_LIB_DIRS})

OPTION(ENABLE_CUDA "" ON)
FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_INCLUDE_DIRS ${CONAN_INCLUDE_DIRS_CUDART})
LIST(APPEND LIBS


        ${CONAN_LIBS_PROTOBUF}
        ${CONAN_LIBS_GTEST}
        ${CONAN_LIBS_WATCHDOG}
        ${CONAN_LIBS_HASP}
        ${CONAN_LIBS_DGMEDIA}
        ${CONAN_LIBS_THPLATEID}
        ${CONAN_LIBS_DGLP}
        ${CONAN_LIBS_DGVEHICLE}
        ${CONAN_LIBS_DGFACE}
        ${CONAN_LIBS_CDNNCAFFE}
        ${CONAN_LIBS_CDNN}
        ${CONAN_LIBS_MXNET}
        ${CONAN_LIBS_DGCAFFE}
        ${CONAN_LIBS_DLIB}
        ${CONAN_LIBS_OPENCV}
        ${CONAN_LIBS_CUDNN}
        ${CONAN_LIBS_CUDART}
        ${CONAN_LIBS})

LIST(APPEND LIBS boost_regex pthread glog gflags leveldb jsoncpp cblas
        crypto b64 snappy lmdb jpeg png GL glut curl uuid freetype
        boost_system boost_filesystem hdf5_hl hdf5 m gstreamer-0.10
        gobject-2.0 gmodule-2.0 gthread-2.0 xml2 glib-2.0 gstapp-0.10 z dl)

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)
