cmake_minimum_required(VERSION 3.25)
project(GMkBPE LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

if(DEFINED CONDA_PREFIX)
    list(PREPEND CMAKE_PREFIX_PATH "${CONDA_PREFIX}")
elseif(DEFINED ENV{CONDA_PREFIX})
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

option(GC_NATIVE "" ON)
option(GC_AVX512 "" OFF)
option(GC_LTO "" ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(TBB QUIET)

if(GC_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT GC_IPO_SUPPORTED OUTPUT GC_IPO_ERROR LANGUAGES CXX)
endif()

set(PROJECT_INCLUDE_DIRS
        "${CMAKE_CURRENT_LIST_DIR}/include"
        "${CMAKE_CURRENT_LIST_DIR}/data/src"
)

set(CUDA_OPENMP_HOST_FLAGS "")
if(OpenMP_CXX_FLAGS)
    string(REPLACE ";" " " OPENMP_CXX_FLAGS_STR "${OpenMP_CXX_FLAGS}")
    separate_arguments(CUDA_OPENMP_HOST_FLAGS NATIVE_COMMAND "${OPENMP_CXX_FLAGS_STR}")
endif()

add_executable(gmkbpe src/main.cu)

target_include_directories(gmkbpe PRIVATE
        ${PROJECT_INCLUDE_DIRS}
)

set_target_properties(gmkbpe PROPERTIES
        CUDA_ARCHITECTURES "80;86;90"
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_LINEINFO ON
)

target_compile_options(gmkbpe PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

foreach(flag IN LISTS CUDA_OPENMP_HOST_FLAGS)
    target_compile_options(gmkbpe PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>
    )
endforeach()

target_link_libraries(gmkbpe PRIVATE
        fmt::fmt
        nlohmann_json::nlohmann_json
        CUDA::cudart
        OpenMP::OpenMP_CXX
)

add_executable(graph_convert preprocess/graph_conver.cpp)

set_target_properties(graph_convert PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

target_include_directories(graph_convert PRIVATE
        ${PROJECT_INCLUDE_DIRS}
)

target_link_libraries(graph_convert PRIVATE
        OpenMP::OpenMP_CXX
)

if(TARGET TBB::tbb)
    target_link_libraries(graph_convert PRIVATE
            TBB::tbb
    )
endif()

if(MSVC)
    target_compile_options(graph_convert PRIVATE
            /W3
            /MP
            /bigobj
            /utf-8
            /Zc:__cplusplus
            $<$<CONFIG:Release>:/O2>
            $<$<CONFIG:Release>:/Oi>
            $<$<CONFIG:Release>:/Ot>
            $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>,$<BOOL:${GC_AVX512}>>:/arch:AVX512>
            $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>,$<NOT:$<BOOL:${GC_AVX512}>>>:/arch:AVX2>
    )
    target_compile_definitions(graph_convert PRIVATE
            _CRT_SECURE_NO_WARNINGS
            NOMINMAX
            WIN32_LEAN_AND_MEAN
    )
else()
    target_compile_options(graph_convert PRIVATE
            -Wall
            -Wextra
            -Wno-unknown-pragmas
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:RelWithDebInfo>:-O2>
            $<$<CONFIG:RelWithDebInfo>:-g>
            $<$<AND:$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>,$<BOOL:${GC_NATIVE}>>:-march=native>
    )
endif()

if(GC_LTO AND GC_IPO_SUPPORTED)
    set_property(TARGET graph_convert PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set_property(TARGET graph_convert PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(graph_convert PRIVATE
            rt
    )
endif()
