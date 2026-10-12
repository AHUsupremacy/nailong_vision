cmake_minimum_required(VERSION 3.16.3)

project(nailong_vision)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks/nailong)

add_executable(nailong src/nailong.cpp)
target_link_libraries(nailong ${OpenCV_LIBS} yaml-cpp Eigen3::Eigen io tools nailong_detector)

# 模型文件拷贝到 build 目录，方便直接运行
add_custom_command(TARGET nailong POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${PROJECT_SOURCE_DIR}/model $<TARGET_FILE_DIR:nailong>/model)
