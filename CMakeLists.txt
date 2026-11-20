cmake_minimum_required(VERSION 4.3)
project(CudaRayTracing C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

add_subdirectory(externals/glfw)
include_directories(externals/glfw/include)
include_directories(externals/imgui)
include_directories(externals/imgui/backends)


find_package(OpenGL REQUIRED)

add_executable(CudaRayTracing
        src/main.cpp
        externals/glad/src/glad.c
        externals/imgui/imgui.cpp
        externals/imgui/imgui_draw.cpp
        externals/imgui/imgui_widgets.cpp
        externals/imgui/imgui_tables.cpp
        externals/imgui/backends/imgui_impl_glfw.cpp
        externals/imgui/backends/imgui_impl_opengl3.cpp
        src/vec3.h
        src/color.h
        src/hittable.h
        src/camera.h
        src/material.h
        src/render.h
        src/render.cu
        src/application.cpp
        src/Application.h
        src/ImguiRenderer.cpp
        src/ImguiRenderer.h
)

target_include_directories(CudaRayTracing PRIVATE externals/glad/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)



target_link_libraries(CudaRayTracing PRIVATE
        glfw

        OpenGL::GL
)


