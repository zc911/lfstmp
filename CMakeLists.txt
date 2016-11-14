cmake_minimum_required(VERSION 2.8)
PROJECT(dgface)
SET(VERSION 0.1.0)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)

conan_basic_setup()

SET(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -std=c++11 -DDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -g  -w")
SET(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -std=c++11 -DNDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -O3 -w")
add_compile_options(-std=c++11)

SET(PLATFORM ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})


#INCLUDE_DIRECTORIES( ${CONAN_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include ${CONAN_INCLUDE_DIRS})
foreach(libdir ${CONAN_LIB_DIRS})
    LIST(APPEND LIBSDIR ${libdir}/${PLATFORM})
endforeach(libdir)
LINK_DIRECTORIES(${LIBSDIR})


LIST(APPEND LIBS glog  boost_filesystem boost_system ${CONAN_LIBS_CDNN}
				${CONAN_LIBS_CDNNCAFFE} 
				${CONAN_LIBS_DLIB} 
				${CONAN_LIBS_GTEST} 
				${CONAN_LIBS_PROTOBUF} 
				${CONAN_LIBS_OPENCV} 
				${CONAN_LIBS_WATCHDOG} 
				${CONAN_LIBS_HASP} 
				${CONAN_LIBS_DGCAFFE} 
				${CONAN_LIBS_CUDART})
LIST(APPEND LIBS pthread cblas crypto )

# TODO make release VERSION to upload the lib and headers to conan server
# add_custom_target(release COMMAND conan upload ${PROJECT_NAME}/${VERSION}@deepglint/demo)

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)
