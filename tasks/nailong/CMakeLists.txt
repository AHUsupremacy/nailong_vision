add_library(nailong_detector STATIC detector.cpp)
target_include_directories(nailong_detector PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(nailong_detector ${OpenCV_LIBS} yaml-cpp)
