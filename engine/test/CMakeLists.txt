enable_testing()
set(PROJECT_TEST_NAME test_${PROJECT_NAME})

file(GLOB_RECURSE TEST_SRC_FILES *.cpp)
add_executable(${PROJECT_TEST_NAME} ${TEST_SRC_FILES})

SET(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -std=c++11 -DDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -g -O3 -w -lpthread -lgtest -lgtest_main -fprofile-arcs -ftest-coverage")
SET(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -std=c++11 -DNDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -O3 -w -lpthread -lgtest -lgtest_main -fprofile-arcs -ftest-coverage")

find_package( GTest )
include_directories( ${GTEST_INCLUDE_DIRS} )
target_link_libraries(${PROJECT_TEST_NAME} ${GTEST_BOTH_LIBRARIES})

target_link_libraries(${PROJECT_TEST_NAME} ${PROJECT_NAME})
target_link_libraries(${PROJECT_TEST_NAME} ${LIBS})

add_test(${PROJECT_TEST_NAME} ${PROJECT_TEST_NAME})