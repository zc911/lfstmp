set(PROJECT_TEST_NAME test_${PROJECT_NAME})

file(GLOB_RECURSE TEST_SRC_FILES *.c *.cc *.cpp)

AUX_SOURCE_DIRECTORY(../../utils/src/ UTILS)
AUX_SOURCE_DIRECTORY(../../engine/src/ ENGINE)
LIST(APPEND ALL_SRC_FILES ${ENGINE} ${UTILS})
LIST(REMOVE_ITEM ALL_SRC_FILES ../src/main.cpp)
add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES} ${ALL_SRC_FILES})

IF (ENABLE_CUDA)
    set(CUDA_NVCC_FLAGS ${CMAKE_CXX_FLAGS_DEBUG};${CUDA_NVCC_FLAGS}; -DUSE_CUDA)

    file(GLOB_RECURSE CUDA_FILES ../src/*.cu)
    set(PROJECT_ENGINE_CUDA_LIB_TEST ${PROJECT_NAME}_cuda_test)
    cuda_add_library(${PROJECT_ENGINE_CUDA_LIB_TEST} ${CUDA_FILES})
ENDIF ()

SET(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -std=c++11 -DDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -g -O3 -w -fprofile-arcs -ftest-coverage")
SET(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -std=c++11 -DNDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -O3 -w -fprofile-arcs -ftest-coverage")


INCLUDE_DIRECTORIES(../../utils/src ../../engine/src ../src ../include ../../engine/include ../../engine/include/dgcaffe ../../engine/include/mxnet ../../engine/include/watchdog ../../engine/include/dgmedia-1.8.0 /usr/local/cuda/include /usr/include/glib-2.0 /usr/lib/x86_64-linux-gnu/glib-2.0/include /usr/include/gstreamer-0.10 /usr/include/glib-2.0 /usr/lib/arm-linux-gnueabihf/glib-2.0/include /usr/include/libxml2 /usr/include/freetype2)

include_directories( ${GTEST_INCLUDE_DIRS} ${UTILS} ${ENGINE})
target_link_libraries(${PROJECT_TEST_NAME} ${GTEST_BOTH_LIBRARIES} ${PROJECT_ENGINE_CUDA_LIB_TEST})

target_link_libraries(${PROJECT_TEST_NAME} ${LIBS})

add_custom_target(test
        COMMAND make
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_TEST_NAME}
        )
