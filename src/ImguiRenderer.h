//
// Created by Renkai on 11/20/2026.
//

#ifndef CUDARAYTRACING_IMGUIRENDERER_H
#define CUDARAYTRACING_IMGUIRENDERER_H
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"


class ImguiRenderer {
public:
    ImguiRenderer(GLFWwindow*window, float main_scale);
    ~ImguiRenderer();
    void render(int pass);
    void newFrame();


private:
    void destroy();
    void init();

    GLFWwindow*window = nullptr;
    float main_scale;

};



#endif //CUDARAYTRACING_IMGUIRENDERER_H
