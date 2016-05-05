enable_testing()
set(PROJECT_TEST_NAME test_${PROJECT_NAME})

file(GLOB_RECURSE TEST_SRC_FILES *.cpp)
add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})

# gtest 
find_package( GTest )
include_directories( ${GTest_INCLUDE_DIRS} )
target_link_libraries(${PROJECT_TEST_NAME} ${GTEST_BOTH_LIBRARIES})

target_link_libraries(${PROJECT_TEST_NAME} ${PROJECT_NAME}) # matrix_util
target_link_libraries(${PROJECT_TEST_NAME} ${LIBS})

add_test(${PROJECT_TEST_NAME} ${PROJECT_TEST_NAME})
