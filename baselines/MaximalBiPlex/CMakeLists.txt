cmake_minimum_required(VERSION 3.29)

project(BiPlex LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -Wno-reorder -O3 -march=native -g -pg")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g -pg -O0")

set(EXECUTABLE_OUTPUT_PATH bin)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(nlohmann_json CONFIG REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/utils)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/biplex)

file(GLOB APP_SOURCES "apps/*.cpp")

foreach(APP_SOURCE ${APP_SOURCES})
    get_filename_component(TARGET_NAME ${APP_SOURCE} NAME_WE)

    add_executable(${TARGET_NAME} ${APP_SOURCE})

    target_link_libraries(${TARGET_NAME} PRIVATE BiPlex Utils nlohmann_json::nlohmann_json)
endforeach()
