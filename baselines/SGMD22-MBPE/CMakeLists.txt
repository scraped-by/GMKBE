cmake_minimum_required(VERSION 3.25)
project(SGMOD22_MBPE)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS "-O3")

find_package(fmt CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include/ ${CMAKE_SOURCE_DIR}/include/cpp-btree-1.0.1)
file(GLOB files ${CMAKE_CURRENT_SOURCE_DIR}/apps/*.cpp)

foreach(file ${files})
    if(${file} STREQUAL  "")
        continue()
    endif ()
    STRING(REGEX REPLACE ".+/(.+)\\..*" "\\1" file_name ${file})

    add_executable(${file_name} ${file})

    target_link_libraries(${file_name} PRIVATE nlohmann_json::nlohmann_json fmt::fmt)
endforeach ()