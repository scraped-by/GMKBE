cmake_minimum_required(VERSION 3.29)


#include_directories(../utils)

add_library(BiPlex biplex.h biplex.cpp)
target_link_libraries(BiPlex Utils)
target_include_directories(BiPlex PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
