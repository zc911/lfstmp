set(DATA_URL "192.168.2.119/unit_test/data")
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
include_directories( ${GTEST_INCLUDE_DIRS} ../src )
target_link_libraries(${PROJECT_TEST_NAME} ${GTEST_BOTH_LIBRARIES} ${PROJECT_ENGINE_CUDA_LIB_TEST})
target_link_libraries(${PROJECT_TEST_NAME} ${LIBS})

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_TEST_NAME} DESTINATION ${PROJECT_SOURCE_DIR}/bin/test )

add_custom_target(test
        COMMAND make -j8 install
        COMMAND if [ ! -d ${PROJECT_SOURCE_DIR}/bin/test/data ]\; then wget -r -R html ${DATA_URL} && mv ${DATA_URL} ${PROJECT_SOURCE_DIR}/bin/test \; fi
        COMMAND cd ${PROJECT_SOURCE_DIR}/bin/test/ && ./${PROJECT_TEST_NAME}
        )

add_custom_target(report
        COMMAND rm -rf ${PROJECT_SOURCE_DIR}/bin/test/unitTestReport || true
        COMMAND mkdir -p ${PROJECT_SOURCE_DIR}/bin/test/unitTestReport
        COMMAND lcov -c -o all.info -d ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND lcov -r all.info "/usr/include*" "/include*" -o result.info
        COMMAND genhtml -o ${PROJECT_SOURCE_DIR}/bin/test/unitTestReport result.info
        COMMAND rm all.info result.info
        )
