cmake_minimum_required(VERSION 3.18)
project(ParticleLifeCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

add_library(particlelife_cuda MODULE main.cpp common.h kernel.cu)

target_include_directories(particlelife_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(particlelife_cuda PRIVATE pybind11::module)

if (WIN32)
    set_target_properties(particlelife_cuda PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(particlelife_cuda PROPERTIES SUFFIX ".so")
endif()

