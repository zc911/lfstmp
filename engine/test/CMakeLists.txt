set(PROJECT_TEST_NAME test_${PROJECT_NAME})

file(GLOB_RECURSE TEST_SRC_FILES ../src/*.c ../src/*.cc ../src/*.cpp *.c *.cc *.cpp)

AUX_SOURCE_DIRECTORY(../../utils/src/ UTILS)
LIST(APPEND ALL_SRC_FILES ${UTILS})
add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES} ${ALL_SRC_FILES})


SET(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -std=c++11 -DDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -g -O3 -w")
SET(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -std=c++11 -DNDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -O3 -w")

IF (ENABLE_CUDA)
    set(CUDA_NVCC_FLAGS ${CMAKE_CXX_FLAGS_DEBUG};${CUDA_NVCC_FLAGS}; -DUSE_CUDA)
    file(GLOB_RECURSE CUDA_FILES ../src/*.cu)
    set(PROJECT_ENGINE_CUDA_LIB_TEST ${PROJECT_NAME}_cuda_test)
    cuda_add_library(${PROJECT_ENGINE_CUDA_LIB_TEST} ${CUDA_FILES})
ENDIF ()

SET(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -std=c++11 -DDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -g -O3 -w -fprofile-arcs -ftest-coverage")
SET(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -std=c++11 -DNDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -O3 -w -fprofile-arcs -ftest-coverage")

message("aaa: " ${CMAKE_CXX_FLAGS_DEBUG})

find_package( GTest )
include_directories( ${GTEST_INCLUDE_DIRS} )
include_directories( ../src )
target_link_libraries(${PROJECT_TEST_NAME} ${GTEST_BOTH_LIBRARIES} ${PROJECT_ENGINE_CUDA_LIB_TEST})
target_link_libraries(${PROJECT_TEST_NAME} ${LIBS})

add_custom_target(test
        COMMAND make
        COMMAND if [ ! -d data ]\; then wget -r -R html http://192.168.2.119/unit_test/data/ && mv 192.168.2.119/unit_test/data . && rm -rf 192.168.2.119 \; fi
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_TEST_NAME}
        )

add_custom_target(report
        COMMAND rm -rf unitTestReport || true
        COMMAND mkdir unitTestReport
        COMMAND lcov -c -o all.info -d ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND lcov -r all.info "/usr/include*" "/include*" -o result.info
        COMMAND genhtml -o ./unitTestReport result.info
        COMMAND rm all.info result.info
        )
