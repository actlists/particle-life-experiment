cmake_minimum_required(VERSION 3.18)

project(ParticleLifeExperiment LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SDL3_DIR "./SDL3-3.2.18/cmake")

# Find SDL3
find_package(SDL3 3.2.18 REQUIRED CONFIG)

# Add executable
add_executable(ParticleLifeExperiment main.cu)

# Include SDL3 headers
target_include_directories(ParticleLifeExperiment PRIVATE ${SDL3_INCLUDE_DIRS})

# Link to SDL3
target_link_libraries(ParticleLifeExperiment PRIVATE SDL3::SDL3)

# Enable CUDA separable compilation (optional, good for bigger projects)
set_target_properties(ParticleLifeExperiment PROPERTIES CUDA_SEPARABLE_COMPILATION ON)