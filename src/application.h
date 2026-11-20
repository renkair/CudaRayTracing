//
// Created by Renkai on 11/20/2026.
//

#ifndef CUDARAYTRACING_APPLICATION_H
#define CUDARAYTRACING_APPLICATION_H
#include "rtweekend.h"
#include <glad/glad.h>
#include "camera.h"
#include "hittable_list.h"
#include "imgui.h"
#include "GLFW/glfw3.h"
#include "ImguiRenderer.h"


class Application {
public:
    void run();
    bool initialize();
    void shutdown();
private:
    static void framebuffer_size_callback(GLFWwindow *window, int width, int height);
    void initWorld(hittable_list& world);

    hittable_list world;
    GLFWwindow*window = nullptr;
    float main_scale = 0;
    ImguiRenderer* imguiRenderer = nullptr;

    GLuint pbo = 0;
    cudaGraphicsResource* pboRes = nullptr;
    color* sumColor = nullptr;
    int pass = 0;
    camera cam;
    unsigned int texture = 0;
    unsigned int shaderProgram = 0;

    // IMGUI State
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    const char* vertexShaderSource = R"(
    #version 330 core
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec2 aTexCoord;

    out vec2 TexCoord;

    void main()
    {
        gl_Position = vec4(aPos, 1.0);
        TexCoord = aTexCoord;
    }
    )";

        const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;

    uniform sampler2D screenTexture;

    void main()
    {
        FragColor = texture(screenTexture, TexCoord);
    }
    )";

    };

    inline unsigned int VAO;
    inline unsigned int VBO;
    inline unsigned int EBO;

#endif //CUDARAYTRACING_APPLICATION_H
