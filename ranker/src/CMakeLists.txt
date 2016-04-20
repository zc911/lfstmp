#include_directories( ${COMMON_INCLUDES} )
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE SRC_FILES *.cpp *.c *.cc)

set(PROJECT_RANKER_LIB ${PROJECT_NAME_STR}_ranker)
add_library(${PROJECT_RANKER_LIB} SHARED ${SRC_FILES})

target_link_libraries(${PROJECT_RANKER_LIB} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${PROJECT_RANKER_LIB} ${FREETYPE_LIBRARIES})
target_link_libraries(${PROJECT_RANKER_LIB} ${JSONCPP_LIBRARIES})
target_link_libraries(${PROJECT_RANKER_LIB} ${GLOG_LIBRARIES})
#target_link_libraries(${PROJECT_RANKER_LIB} ${GRPC_LIBRARIES})
target_link_libraries(${PROJECT_RANKER_LIB} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_RANKER_LIB} ${PROTOBUF_LIBRARY})
target_link_libraries(${PROJECT_RANKER_LIB} ${DLIB_LIBRARY})
target_link_libraries(${PROJECT_RANKER_LIB} ${Caffe_LIBRARIES})
target_link_libraries(${PROJECT_RANKER_LIB} ${PROJECT_NAME_STR}_utils)
