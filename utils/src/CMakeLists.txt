
#include_directories( ${COMMON_INCLUDES} )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE SRC_FILES *.cpp *.c)

add_library(${PROJECT_NAME_PREFIX} SHARED ${SRC_FILES})
target_link_libraries(${PROJECT_NAME_PREFIX} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${PROJECT_NAME_PREFIX} ${FREETYPE_LIBRARIES})
target_link_libraries(${PROJECT_NAME_PREFIX} ${JSONCPP_LIBRARIES})
target_link_libraries(${PROJECT_NAME_PREFIX} ${GLOG_LIBRARIES})
target_link_libraries(${PROJECT_NAME_PREFIX} ${OpenCV_LIBS})
