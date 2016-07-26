set(PROJECT_TEST_NAME test_${PROJECT_NAME})

file(GLOB_RECURSE TEST_SRC_FILES  *.cpp)

AUX_SOURCE_DIRECTORY(../../utils/src/ UTILS)
LIST(APPEND ALL_SRC_FILES ${UTILS})

AUX_SOURCE_DIRECTORY(../src/processor SRC_PRO)
AUX_SOURCE_DIRECTORY(../src/processor/tbb SRC_PRO_TBB)
AUX_SOURCE_DIRECTORY(../src/alg SRC_ALG)
AUX_SOURCE_DIRECTORY(../src/alg/classification SRC_CLASSIFICATION)
AUX_SOURCE_DIRECTORY(../src/alg/detector SRC_DECTOR)
AUX_SOURCE_DIRECTORY(../src/alg/feature SRC_FEATURE)
AUX_SOURCE_DIRECTORY(../src/alg/plate SRC_PLATE)
AUX_SOURCE_DIRECTORY(../src/alg/tracking SRC_TRACKING)
AUX_SOURCE_DIRECTORY(../src/engine SRC_ENGINE)
AUX_SOURCE_DIRECTORY(../src/io SRC_IO)
AUX_SOURCE_DIRECTORY(../src/log SRC_LOG)
AUX_SOURCE_DIRECTORY(../src/network SRC_NETWORK)
AUX_SOURCE_DIRECTORY(../src/model SRC_MODEL)
AUX_SOURCE_DIRECTORY(../src/security SRC_SECURITY)
AUX_SOURCE_DIRECTORY(../src/util SRC_UTIL)
AUX_SOURCE_DIRECTORY(../src/vis SRC_VIS)

LIST(APPEND ALL_SRC_FILES ${SRC_ALG} ${SRC_PRO_TBB} ${SRC_PRO} ${SRC_PLATE} ${SRC_MODEL} ${SRC_CLASSIFICATION} ${SRC_DECTOR})
LIST(APPEND ALL_SRC_FILES ${SRC_ENGINE} ${SRC_FEATURE} ${SRC_NETWORK} ${SRC_TRACKING} ${SRC_IO} ${SRC_LOG} ${SRC_MODEL})
LIST(APPEND ALL_SRC_FILES ${SRC_SECURITY} ${SRC_UTIL} ${SRC_VIS})
LIST(REMOVE_ITEM ALL_SRC_FILES ../src/alg/feature/car_matcher.cpp)
LIST(REMOVE_ITEM ALL_SRC_FILES ../src/processor/car_rank_processor.cpp)
LIST(REMOVE_ITEM ALL_SRC_FILES ../src/engine/rank_engine.cpp)

add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES} ${ALL_SRC_FILES})

SET(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -std=c++11 -DDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -g -O3 -w -lpthread -fprofile-arcs -ftest-coverage")
SET(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -std=c++11 -DNDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -O3 -w -lpthread -fprofile-arcs -ftest-coverage")

find_package( GTest )
include_directories( ${GTEST_INCLUDE_DIRS} )
include_directories( ../src )
target_link_libraries(${PROJECT_TEST_NAME} ${GTEST_BOTH_LIBRARIES})

#target_link_libraries(${PROJECT_TEST_NAME} ${PROJECT_NAME})
target_link_libraries(${PROJECT_TEST_NAME} ${LIBS})

add_custom_target(test
        COMMAND make
        COMMAND if [ ! -d data ]\; then wget -r -R html http://192.168.2.119/unit_test/data/ && mv 192.168.2.119/unit_test/data . && rm -rf 192.168.2.119 \; fi
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_TEST_NAME}
#        COMMAND rm -rf unitTestReport
#        COMMAND mkdir unitTestReport
#        COMMAND lcov -c -o all.info -d ${CMAKE_CURRENT_BINARY_DIR}
#        COMMAND lcov -r all.info "/usr/include*" "/include*" -o result.info
#        COMMAND genhtml -o ./unitTestReport result.info
#        COMMAND rm all.info result.info
        )

add_custom_target(report
        COMMAND rm -rf unitTestReport
        COMMAND mkdir unitTestReport
        COMMAND lcov -c -o all.info -d ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND lcov -r all.info "/usr/include*" "/include*" -o result.info
        COMMAND genhtml -o ./unitTestReport result.info
        COMMAND rm all.info result.info
        )
