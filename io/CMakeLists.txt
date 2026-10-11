add_library(io STATIC camera.cpp usbcamera.cpp hikrobot/hikrobot.cpp)
target_include_directories(io PUBLIC
  ${PROJECT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/hikrobot/include)
target_link_libraries(io ${OpenCV_LIBS} yaml-cpp usb-1.0)

# 海康 MVS SDK：按架构选 .so
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(HIKROBOT_LIB ${CMAKE_CURRENT_SOURCE_DIR}/hikrobot/lib/arm64/libMvCameraControl.so)
else()
  set(HIKROBOT_LIB ${CMAKE_CURRENT_SOURCE_DIR}/hikrobot/lib/amd64/libMvCameraControl.so)
endif()
target_link_libraries(io ${HIKROBOT_LIB})

# 运行时能找到 .so：拷到 build 并加 rpath
add_custom_command(TARGET io POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${HIKROBOT_LIB} ${CMAKE_BINARY_DIR})
set_target_properties(io PROPERTIES BUILD_RPATH ${CMAKE_BINARY_DIR})
