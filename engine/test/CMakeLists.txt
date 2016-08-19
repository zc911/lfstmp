set(BASE_URL "192.168.2.119/matrix")
set(RELEASE_URL ${BASE_URL}"/Linux-x86_64")
set(DATA_URL ${BASE_URL}"/unit_test/data")
set(PROJECT_TEST_NAME test_${PROJECT_NAME})

file(GLOB_RECURSE TEST_SRC_FILES ../src/*.c ../src/*.cc ../src/*.cpp *.c *.cc *.cpp)
#file(GLOB_RECURSE TEST_SRC_FILES ../src/*.c ../src/*.cc ../src/*.cpp
#        processor/frame_batch_helper.cpp processor/vehicle_processor_head.cpp
#        processor/vehicle_marker_classifier_processor_test.cpp)

IF(CMAKE_BUILD_TYPE STREQUAL "Debug")
    ADD_DEFINITIONS( -DUNENCRYPTMODEL )
ENDIF()

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

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_TEST_NAME} DESTINATION ${PROJECT_SOURCE_DIR}/bin/test )

add_custom_target(test
        COMMAND make -j8 install
        COMMAND wget -rq -R html ${RELEASE_URL}/version -O version
        COMMAND echo -n ${RELEASE_URL} > task.txt
        COMMAND echo -n "/release/data/matrix_apps_data_" >> task.txt
        COMMAND echo -n `cat version` >> task.txt
        COMMAND echo ".tar" >> task.txt
        COMMAND if [ ! -d ${PROJECT_SOURCE_DIR}/bin/test/data ]\;
            then echo "Loading models..."
            && wget -rq -R html ${DATA_URL}/
            && wget -rq -R html ${RELEASE_URL}/internal/data/
            && wget -rqi task.txt -O matrix_models.tar
            && tar -xf matrix_models.tar
            && mv ${DATA_URL} ${PROJECT_SOURCE_DIR}/bin/test
            && mv ${RELEASE_URL}/internal/data/* ${PROJECT_SOURCE_DIR}/bin/test/data
            && mv bin/Release/data/1 ${PROJECT_SOURCE_DIR}/bin/test/data/1
            && rm -rf bin matrix_models.tar \; fi
        COMMAND diff ${PROJECT_SOURCE_DIR}/bin/test/data/version version > /dev/null
            || (echo "Updating models..."
            && rm -rf ${PROJECT_SOURCE_DIR}/bin/test/data
            && wget -rq -R html ${DATA_URL}/
            && wget -rq -R html ${RELEASE_URL}/internal/data/
            && wget -rqi task.txt -O matrix_models.tar
            && tar -xf matrix_models.tar
            && mv ${DATA_URL} ${PROJECT_SOURCE_DIR}/bin/test
            && mv ${RELEASE_URL}/internal/data/* ${PROJECT_SOURCE_DIR}/bin/test/data
            && mv bin/Release/data/1 ${PROJECT_SOURCE_DIR}/bin/test/data/1
            && rm -rf bin matrix_models.tar
            && mv version ${PROJECT_SOURCE_DIR}/bin/test/data/version)
        COMMAND rm -rf ${PROJECT_SOURCE_DIR}/bin/test/data/testimg || true
        COMMAND echo "Loading data..." && wget -rq -R html ${DATA_URL}/testimg/
            && mv ${DATA_URL}/testimg ${PROJECT_SOURCE_DIR}/bin/test/data/testimg
            && rm -rf ${BASE_URL} \; fi
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
