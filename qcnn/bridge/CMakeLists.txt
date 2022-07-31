cmake_minimum_required(VERSION 3.10)
project(bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

#add_executable(angle angle.cpp)
add_library(angle SHARED angle.cpp)
#add_executable(amplitude amplitude.cpp)
add_library(amplitude SHARED amplitude.cpp)

find_package(OpenMP)

target_link_libraries (angle OpenMP::OpenMP_CXX)
target_link_libraries (amplitude OpenMP::OpenMP_CXX)

include_directories(../../../qrack/include)
find_library(QRACK qrack ../../../qrack)
target_link_libraries(angle ${QRACK})
target_link_libraries(amplitude ${QRACK})

set_target_properties(angle PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(amplitude PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})