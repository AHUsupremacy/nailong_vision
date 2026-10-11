add_library(io STATIC camera.cpp)
target_include_directories(io PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(io ${OpenCV_LIBS} yaml-cpp)
