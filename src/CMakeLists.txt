FILE(GLOB_RECURSE ALL_SRC_FILES *.cpp *.c *.cc)
ADD_LIBRARY(${PROJECT_NAME} SHARED ${ALL_SRC_FILES})

INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/dist/lib/)
#INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/include DESTINATION ${PROJECT_SOURCE_DIR}/release)

LIST(APPEND RELEASE_H common.h dgface_utils.h detector.h alignment.h recognition.h transformation.h verification.h quality.h tracking.h database.h)
foreach(releasehead ${RELEASE_H})
	LIST(APPEND FULL_RELEASE_H ${PROJECT_SOURCE_DIR}/include/${releasehead})
endforeach(releasehead)
INSTALL(FILES ${FULL_RELEASE_H} DESTINATION ${PROJECT_SOURCE_DIR}/dist/include/dgface)
add_custom_command(
		TARGET ${PROJECT_NAME} POST_BUILD
		COMMAND rm -rf ${PROJECT_SOURCE_DIR}/release/include ${PROJECT_SOURCE_DIR}/release/lib
		COMMAND ln -s ${PROJECT_SOURCE_DIR}/dist/include ${PROJECT_SOURCE_DIR}/release/include
		COMMAND ln -s ${PROJECT_SOURCE_DIR}/dist/lib ${PROJECT_SOURCE_DIR}/release/lib
)