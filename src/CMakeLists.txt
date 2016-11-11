FILE(GLOB_RECURSE ALL_SRC_FILES *.cpp *.c *.cc)
ADD_LIBRARY(${PROJECT_NAME} SHARED ${ALL_SRC_FILES})

INSTALL(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/release/${PLATFORM})
#INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/include DESTINATION ${PROJECT_SOURCE_DIR}/release)

LIST(APPEND RELEASE_H common.h config.h dgface_utils.h detector.h alignment.h recognition.h transformation.h verification.h quality.h tracking.h database.h)
foreach(releasehead ${RELEASE_H})
	LIST(APPEND FULL_RELEASE_H ${PROJECT_SOURCE_DIR}/include/${releasehead})
endforeach(releasehead)
INSTALL(FILES ${FULL_RELEASE_H} DESTINATION ${PROJECT_SOURCE_DIR}/release/include)
