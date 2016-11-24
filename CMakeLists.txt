cmake_minimum_required(VERSION 2.8)
#PROJECT(matrix_all)

INCLUDE_DIRECTORIES(apps/include apps/src apps/include/matrix_engine utils/dist/${PLATFORM}/include)

ADD_SUBDIRECTORY(engine)

ADD_SUBDIRECTORY(apps)
