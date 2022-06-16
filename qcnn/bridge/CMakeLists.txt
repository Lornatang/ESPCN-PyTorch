cmake_minimum_required(VERSION 3.10)
project(bridge)

set(CMAKE_CXX_STANDARD 17)

add_executable(angle angle.cpp)
#add_library(angle SHARED angle.cpp)
#add_executable(amplitude amplitude.cpp)
add_library(amplitude SHARED amplitude.cpp)

find_package (OpenCL)
target_compile_definitions (angle PUBLIC CL_HPP_TARGET_OPENCL_VERSION=200)
target_compile_definitions (amplitude PUBLIC CL_HPP_TARGET_OPENCL_VERSION=200)
target_compile_definitions (angle PUBLIC CL_HPP_MINIMUM_OPENCL_VERSION=110)
target_compile_definitions (amplitude PUBLIC CL_HPP_MINIMUM_OPENCL_VERSION=110)
set(QRACK_OpenCL_LIBRARIES ${OpenCL_LIBRARIES})
set(QRACK_OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS})
set(QRACK_OpenCL_COMPILATION_OPTIONS ${OpenCL_COMPILATION_OPTIONS})
target_include_directories (angle PUBLIC ${PROJECT_BINARY_DIR} ${QRACK_OpenCL_INCLUDE_DIRS})
target_include_directories (amplitude PUBLIC ${PROJECT_BINARY_DIR} ${QRACK_OpenCL_INCLUDE_DIRS})
target_compile_options (angle PUBLIC ${QRACK_OpenCL_COMPILATION_OPTIONS})
target_compile_options (amplitude PUBLIC ${QRACK_OpenCL_COMPILATION_OPTIONS})

find_package(OpenMP)

target_link_libraries (angle ${QRACK_OpenCL_LIBRARIES} OpenMP::OpenMP_CXX)
target_link_libraries (amplitude ${QRACK_OpenCL_LIBRARIES} OpenMP::OpenMP_CXX)

include_directories(../../../qrack/include)
find_library(QRACK qrack ../../../qrack)
target_link_libraries(angle ${QRACK})
target_link_libraries(amplitude ${QRACK})

set_target_properties(angle PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(amplitude PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})