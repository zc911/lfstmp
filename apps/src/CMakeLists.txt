AUX_SOURCE_DIRECTORY(./grpc SRC_GRPC)
AUX_SOURCE_DIRECTORY(./model SRC_MODEL)
AUX_SOURCE_DIRECTORY(./restful SRC_RESTFUL)
AUX_SOURCE_DIRECTORY(./service SRC_SERVICE)

LIST(APPEND SRC_LIST ${SRC_GRPC} ${SRC_MODEL} ${SRC_RESTFUL} ${SRC_SERVICE})

set(GRPC_PROJECT ${PROJECT_NAME_STR}_grpc)
add_executable(${GRPC_PROJECT} ${SRC_LIST} grpc_main.cpp)

target_link_libraries(${GRPC_PROJECT} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${GRPC_PROJECT} ${FREETYPE_LIBRARIES})
target_link_libraries(${GRPC_PROJECT} ${JSONCPP_LIBRARIES})
target_link_libraries(${GRPC_PROJECT} ${GLOG_LIBRARIES})
target_link_libraries(${GRPC_PROJECT} ${GRPC_LIBRARIES})
target_link_libraries(${GRPC_PROJECT} ${OpenCV_LIBS})
target_link_libraries(${GRPC_PROJECT} ${PROTOBUF_LIBRARY})
target_link_libraries(${GRPC_PROJECT} ${DLIB_LIBRARY})
target_link_libraries(${GRPC_PROJECT} ${Caffe_DEP_LIBRARIES})
target_link_libraries(${GRPC_PROJECT} ${PROJECT_NAME_STR}_utils)
target_link_libraries(${GRPC_PROJECT} ${PROJECT_NAME_STR}_engine)

set(RESTFUL_PROJECT ${PROJECT_NAME_STR}_restful)
add_executable(${RESTFUL_PROJECT} ${SRC_LIST} restful_main.cpp)


target_link_libraries(${RESTFUL_PROJECT} ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(${RESTFUL_PROJECT} ${FREETYPE_LIBRARIES})
target_link_libraries(${RESTFUL_PROJECT} ${JSONCPP_LIBRARIES})
target_link_libraries(${RESTFUL_PROJECT} ${GLOG_LIBRARIES})
target_link_libraries(${RESTFUL_PROJECT} ${GRPC_LIBRARIES})
target_link_libraries(${RESTFUL_PROJECT} ${OpenCV_LIBS})
target_link_libraries(${RESTFUL_PROJECT} ${PROTOBUF_LIBRARY})
target_link_libraries(${RESTFUL_PROJECT} ${DLIB_LIBRARY})
target_link_libraries(${RESTFUL_PROJECT} ${Caffe_DEP_LIBRARIES})
target_link_libraries(${RESTFUL_PROJECT} ${PROJECT_NAME_STR}_utils)
target_link_libraries(${RESTFUL_PROJECT} ${PROJECT_NAME_STR}_engine)

#INSTALL(TARGETS witness RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE})
