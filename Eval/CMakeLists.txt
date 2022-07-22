cmake_minimum_required(VERSION 3.16.3)

project(NiftiEval)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(${PROJECT_NAME} NiftiEval.cxx)
target_link_libraries(${PROJECT_NAME} ${ITK_LIBRARIES})

install(TARGETS ${PROJECT_NAME}
  DESTINATION bin/
  COMPONENT Runtime
  )