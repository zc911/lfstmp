
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE SRC_FILES *.cpp *.c *.cc)
set(PROJECT_GRPC_WRAP ${PROJECT_NAME_STR}_grpc)
add_executable(${PROJECT_GRPC_WRAP} ${SRC_FILES})

target_link_libraries(${PROJECT_GRPC_WRAP} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${PROJECT_GRPC_WRAP} ${FREETYPE_LIBRARIES})
target_link_libraries(${PROJECT_GRPC_WRAP} ${JSONCPP_LIBRARIES})
target_link_libraries(${PROJECT_GRPC_WRAP} ${GLOG_LIBRARIES})
target_link_libraries(${PROJECT_GRPC_WRAP} ${GRPC_LIBRARIES})
target_link_libraries(${PROJECT_GRPC_WRAP} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_GRPC_WRAP} ${PROTOBUF_LIBRARY})
target_link_libraries(${PROJECT_GRPC_WRAP} ${DLIB_LIBRARY})
target_link_libraries(${PROJECT_GRPC_WRAP} ${Caffe_LIBRARIES})
target_link_libraries(${PROJECT_GRPC_WRAP} ${PROJECT_NAME_STR}_utils)
target_link_libraries(${PROJECT_GRPC_WRAP} ${PROJECT_NAME_STR}_engine)