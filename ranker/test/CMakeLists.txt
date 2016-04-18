enable_testing()
set(PROJECT_TEST_NAME test_${PROJECT_NAME_PREFIX})

file(GLOB_RECURSE TEST_SRC_FILES *.cpp)
add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})

find_package( GTest )
include_directories( ${GTest_INCLUDE_DIRS} )
target_link_libraries(${PROJECT_TEST_NAME} ${GTEST_BOTH_LIBRARIES})

target_link_libraries(${PROJECT_TEST_NAME} ${CMAKE_THREAD_LIBS_INIT})

target_link_libraries(${PROJECT_TEST_NAME} ${PROJECT_NAME_STR}_utils)
target_link_libraries(${PROJECT_TEST_NAME} ${PROJECT_NAME_STR}_ranker)
add_test(${PROJECT_TEST_NAME} ${PROJECT_TEST_NAME})