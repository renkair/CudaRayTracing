cmake_minimum_required(VERSION 4.3)
project(CudaRayTracing C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(externals/glfw)
include_directories(externals/glfw/include)


find_package(OpenGL REQUIRED)

add_executable(CudaRayTracing
        src/main.cpp
        externals/glad/src/glad.c
        src/vec3.h
        src/color.h
        src/hittable.h
        src/camera.h
        src/material.h
        src/render.h
        src/render.cu
        src/material.cuh
)

target_include_directories(CudaRayTracing PRIVATE externals/glad/include)
target_include_directories(CudaRayTracing PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(CudaRayTracing PRIVATE
        glfw
        OpenGL::GL
)

set_target_properties(CudaRayTracing PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
