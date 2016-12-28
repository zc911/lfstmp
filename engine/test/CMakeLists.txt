set(BASE_URL "192.168.2.119/matrix")
set(RELEASE_URL ${BASE_URL}"/Linux-x86_64")
set(DATA_URL ${BASE_URL}"/unit_test/data")
set(PROJECT_TEST_NAME test_${PROJECT_NAME})

file(GLOB_RECURSE TEST_SRC_FILES ../src/*.c ../src/*.cc ../src/*.cpp *.c *.cc *.cpp)
#file(GLOB_RECURSE TEST_SRC_FILES ../src/*.c ../src/*.cc ../src/*.cpp
#        processor/frame_batch_helper.cpp processor/vehicle_processor_head.cpp
#        processor/pedestrian_classifier_processor_test.cpp)

IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
    ADD_DEFINITIONS( -DUNENCRYPTMODEL )
ENDIF()

SET(CMAKE_INSTALL_RPATH ./libs)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

AUX_SOURCE_DIRECTORY(../../utils/src/ UTILS)
LIST(APPEND ALL_SRC_FILES ${UTILS})
add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES} ${ALL_SRC_FILES})

IF (ENABLE_CUDA)
    set(CUDA_NVCC_FLAGS ${CMAKE_CXX_FLAGS_DEBUG};${CUDA_NVCC_FLAGS}; -DUSE_CUDA)
    file(GLOB_RECURSE CUDA_FILES ../src/*.cu)
    set(PROJECT_ENGINE_CUDA_LIB_TEST ${PROJECT_NAME}_cuda_test)
    cuda_add_library(${PROJECT_ENGINE_CUDA_LIB_TEST} ${CUDA_FILES})
ENDIF ()

SET(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -std=c++11 -DDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -g -w -fprofile-arcs -ftest-coverage")
SET(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -std=c++11 -DNDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -O3 -w -fprofile-arcs -ftest-coverage")

include_directories( ../src )
target_link_libraries(${PROJECT_TEST_NAME} ${LIBS} ${PROJECT_ENGINE_CUDA_LIB_TEST})

set(PROJECT_TEST_Model ${PROJECT_SOURCE_DIR}/../data)
set(PROJECT_TEST_CONFIG ${PROJECT_SOURCE_DIR}/../apps/src/config/config_example.json)
install(TARGETS ${PROJECT_TEST_NAME} RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin/test )
FILE(GLOB_RECURSE RUNTIME_LIBS ${CONAN_LIB_DIRS_DGCAFFE}/${PLATFORM}/* ${CONAN_LIB_DIRS_MXNET}/${PLATFORM}/* ${CONAN_LIB_DIRS_GRPC}/${PLATFORM}/* ${CONAN_LIB_DIRS_GRPCPP}/${PLATFORM}/* ${CONAN_LIB_DIRS_PROTOBUF}/${PLATFORM}/* ${CONAN_LIB_DIRS_DGVEHICLE}/${PLATFORM}/* ${CONAN_LIB_DIRS_DGLP}/${PLATFORM}/* ${CONAN_LIB_DIRS_MXNET}/${PLATFORM}/* ${CONAN_LIB_DIRS_CUDNN}/${PLATFORM}/*)
install(FILES ${RUNTIME_LIBS} DESTINATION ${PROJECT_SOURCE_DIR}/bin/test/libs)
install(FILES ${PROJECT_TEST_CONFIG} DESTINATION ${PROJECT_SOURCE_DIR}/bin/test)
install(DIRECTORY ${PROJECT_TEST_Model} DESTINATION ${PROJECT_SOURCE_DIR}/bin/test)

add_custom_target(test
        COMMAND make -j8 install
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
