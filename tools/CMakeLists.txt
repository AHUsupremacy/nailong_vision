add_library(tools STATIC exiter.cpp math_tools.cpp crc.cpp)
target_include_directories(tools PUBLIC ${PROJECT_SOURCE_DIR})
