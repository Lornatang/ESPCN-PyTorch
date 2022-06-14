cmake_minimum_required(VERSION 3.10)
project(bridge)

set(CMAKE_CXX_STANDARD 17)

#add_executable(bridge main.cpp)
add_library(bridge SHARED main.cpp)

find_package (OpenCL)
target_compile_definitions (bridge PUBLIC CL_HPP_TARGET_OPENCL_VERSION=200)
target_compile_definitions (bridge PUBLIC CL_HPP_MINIMUM_OPENCL_VERSION=110)
set(QRACK_OpenCL_LIBRARIES ${OpenCL_LIBRARIES})
set(QRACK_OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS})
set(QRACK_OpenCL_COMPILATION_OPTIONS ${OpenCL_COMPILATION_OPTIONS})
target_include_directories (bridge PUBLIC ${PROJECT_BINARY_DIR} ${QRACK_OpenCL_INCLUDE_DIRS})
target_compile_options (bridge PUBLIC ${QRACK_OpenCL_COMPILATION_OPTIONS})

find_package(OpenMP)

target_link_libraries (bridge
        ${QRACK_OpenCL_LIBRARIES}
        OpenMP::OpenMP_CXX)

include_directories(../../../qrack/include)
find_library(QRACK qrack ../../../qrack)
target_link_libraries(bridge ${QRACK})

