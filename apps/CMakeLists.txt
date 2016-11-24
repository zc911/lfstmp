cmake_minimum_required(VERSION 2.8)
PROJECT(matrix_apps)

IF(CMAKE_BUILD_TYPE)
    if (DEFINED CODEVERSION AND DEFINED MODELVERSION)
        ADD_DEFINITIONS( -DCODEVERSION="${CODEVERSION}")
        ADD_DEFINITIONS( -DMODELVERSION="${MODELVERSION}")
    elseif(DEFINED DEFAULTVERSION)
        if (DEFINED CODEVERSION AND DEFINED MODELVERSION)
        else()
            set(CODEVERSION 1.0.0)
            set(MODELVERSION 1.9)
        endif()
        ADD_DEFINITIONS( -DCODEVERSION="${CODEVERSION}")
        ADD_DEFINITIONS( -DMODELVERSION="${MODELVERSION}")
        set(DEFAULTVERSION 0)
        unset(DEFAULTVERSION CACHE)
    else()
        message(FATAL_ERROR "lack of -DCODEVERSION=\"version\" -DMODELVERSION=\"version\"
     or -DDEFAULTVERSION=1")
    endif()

    message(STATUS "code version is ${CODEVERSION}")
    message(STATUS "model version is ${MODELVERSION}")
ELSE()
    SET(CMAKE_BUILD_TYPE "Debug")
ENDIF()

SET(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -std=c++11 -DDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -g -w")
SET(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -std=c++11 -DNDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -O3 -w")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

SET(PLATFORM ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})
SET(INCLUDE_PATH ${PROJECT_SOURCE_DIR}/../include)
SET(LIBS_PATH ${PROJECT_SOURCE_DIR}/../lib)

INCLUDE_DIRECTORIES(${CONAN_INCLUDE_DIRS} ${INCLUDE_PATH} src ${INCLUDE_PATH}/matrix_engine ${INCLUDE_PATH}/matrix_util ../imodel/src/cpp)
foreach(libdir ${CONAN_LIB_DIRS})
    LIST(APPEND LIBSDIR ${libdir}/${PLATFORM})
endforeach(libdir)
LINK_DIRECTORIES(${LIBSDIR})

OPTION(ENABLE_CUDA "" ON)
#FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(Boost COMPONENTS regex system filesystem coroutine context REQUIRED)

FIND_LIBRARY(LIB_MATRIX_UTILS NAMES matrix_util PATHS ${LIBS_PATH}/matrix_util/${PLATFORM})
FIND_LIBRARY(LIB_MATRIX matrix PATHS ${LIBS_PATH}/matrix_engine/${PLATFORM})
FIND_LIBRARY(LIB_MATRIX_CUDA matrix_cuda PATHS ${LIBS_PATH}/matrix_engine/${PLATFORM})

LIST(APPEND LIBS ${CONAN_LIBS_DGVEHICLE}
                 ${CONAN_LIBS_PBJSON}
                 ${CONAN_LIBS_PROTOBUF}
                 ${CONAN_LIBS_OPENCV}
                 ${CONAN_LIBS_WATCHDOG}
                 ${CONAN_LIBS_HASP}
                 ${CONAN_LIBS_DGCAFFE}
                 ${CONAN_LIBS_MXNET}            
                 ${CONAN_LIBS_GTEST}
                 ${CONAN_LIBS_CUDNN}
                 ${CONAN_LIBS_THPLATEID}
                 ${CONAN_LIBS_CUDART} 
                 ${CONAN_LIBS_CUDA})

LIST(APPEND LIBS ${LIB_MATRIX} ${LIB_MATRIX_CUDA} ${LIB_MATRIX_UTILS} ${CONAN_LIBS_DLIB} ${CONAN_LIBS_OPENCV} ${CONAN_LIBS_DGLP})
LIST(APPEND LIBS ${Boost_LIBRARIES} ${CONAN_LIBS_GRPC} ${CONAN_LIBS_GRPCPP})
LIST(APPEND LIBS glog gflags leveldb jsoncpp b64 snappy lmdb jpeg png GL glut curl uuid
freetype hdf5_hl hdf5 m crypto gstreamer-0.10 gobject-2.0 gmodule-2.0 gthread-2.0 xml2 glib-2.0 gstapp-0.10 z dl pthread)

ADD_SUBDIRECTORY(src)
#ADD_SUBDIRECTORY(test)
#ADD_SUBDIRECTORY(tools)