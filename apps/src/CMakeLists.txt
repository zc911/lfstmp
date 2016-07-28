file(GLOB_RECURSE SRC_FILES *.cpp *.c *.cc)
file(GLOB_RECURSE SRC_FILES_grpc "../../service/src/cpp/*.cc")

LIST(APPEND SRC_FILES ${SRC_FILES_grpc})

add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} ${LIBS})

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin/${CMAKE_BUILD_TYPE}/)
