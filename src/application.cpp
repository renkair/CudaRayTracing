//
// Created by Renkai on 11/20/2026.
//

#include "application.h"

#include <cuda_gl_interop.h>

#include "render.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"


bool Application::initialize() {
    // World
    initWorld(world);
    world.prepareSceneData();


    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 1280;
    cam.samples_per_pixel = 500;
    cam.max_depth         = 50;
    cam.vfov     = 20;
    cam.lookfrom = point3(13,2,3);
    cam.lookat   = point3(0,0,0);
    cam.vup      = vec3(0,1,0);
    cam.defocus_angle = 0.6;
    cam.focus_dist    = 10.0;
    //cam.render(world);
    cam.initialize();
    cam.pixels.resize(cam.image_width * cam.image_height * 3);


    initDeviceData(world.sceneData_h, cam.cameraData_h, cam.image_width, cam.image_height);

    /* Initialize the library */
    if (!glfwInit())
        return false;

   main_scale = ImGui_ImplGlfw_GetContentScaleForMonitor(glfwGetPrimaryMonitor()); // Valid on GLFW 3.3+ only

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(cam.image_width, cam.image_height, "CudaRayTracing", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return false;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    imguiRenderer = new ImguiRenderer(window, main_scale);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return false;
    }

    //glViewport(0, 0, 800, 600);
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    glfwSetFramebufferSizeCallback(window, Application::framebuffer_size_callback);


    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB8,
        cam.image_width,
        cam.image_height,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        cam.pixels.data()
    );


    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, cam.image_width*cam.image_height*3, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);


    cudaError_t err = cudaGraphicsGLRegisterBuffer(&pboRes, pbo, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cout << "Failed to register buffer resource" << std::endl;
    }


    float vertices[] = {
        1.0f,  1.0f, 0.0f, 1.0f, 0.0f, // top right
        1.0f, -1.0f, 0.0f, 1.0f, 1.0f, // bottom right
       -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, // bottom left
       -1.0f,  1.0f, 0.0f, 0.0f, 0.0f  // top left
   };
    unsigned int indices[] = {  // note that we start from 0!
        0, 1, 3,   // first triangle
        1, 2, 3    // second triangle
    };


    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    // 1. bind Vertex Array Object
    glBindVertexArray(VAO);

    // 2. copy our vertices array in a vertex buffer for OpenGL to use
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    //3. copy our index array in a element buffer for OpenGL to use
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // 4. then set the vertex attributes pointers
    // Position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Texture Coordinates
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    /// shaders
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // 2. use our shader program when we want to render an object
    glUseProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);



    cudaMalloc((void**)&sumColor, cam.image_width*cam.image_height*sizeof(color));

    return true;
}


void Application::run() {
    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {

        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0)
        {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }



        cudaError_t err = cudaGraphicsMapResources(1, &pboRes, 0);
        if (err != cudaSuccess) {
            std::cout << "Failed to map resources" << std::endl;
        }
        void* devPtr = nullptr;
        size_t nbytes = 0;
        err = cudaGraphicsResourceGetMappedPointer(&devPtr, &nbytes, pboRes);
        if (err != cudaSuccess) {
            std::cout << "Get mapped ptr failed" << std::endl;
        }
        /* Render here */

        renderPass(sumColor, static_cast<unsigned char*>(devPtr), ++pass, cam.image_width, cam.image_height);

        cudaDeviceSynchronize();
        cudaGraphicsUnmapResources(1, &pboRes, 0);

        glBindTexture(GL_TEXTURE_2D, texture);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cam.image_width, cam.image_height, GL_RGB, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);


        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);



        imguiRenderer->render(pass);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();



    }
}




void Application::shutdown() {
    // Cleanup
    cudaDestroy();
    glfwTerminate();
}

void Application::framebuffer_size_callback(GLFWwindow *window, int width, int height) {
    glViewport(0.0f, 0.0f, width, height);
}

void Application::initWorld(hittable_list &world) {
    auto ground_material = make_shared<lambertian>(color(0.5f, 0.5f, 0.5f));
    world.add(make_shared<sphere>(point3(0.0f,-1000.0f, 0.0f), 1000.0f, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = (float)random_float();
            point3 center(a + 0.9f*random_float(), 0.2f, b + 0.9f*random_float());

            if ((center - point3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2f, sphere_material));
                } else if (choose_mat < 0.95f) {
                    // metal
                    auto albedo = color::random(0.5f, 1.0f);
                    float fuzz = random_float(0.0f, 0.5f);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2f, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<dielectric>(1.5f);
                    world.add(make_shared<sphere>(center, 0.2f, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5f);
    world.add(make_shared<sphere>(point3(0.0f, 1.0f, 0.0f), 1.0f, material1));

    auto material2 = make_shared<lambertian>(color(0.4f, 0.2f, 0.1f));
    world.add(make_shared<sphere>(point3(-4.0f, 1.0f, 0.0f), 1.0f, material2));

    auto material3 = make_shared<metal>(color(0.7f, 0.6f, 0.5f), 0.0f);
    world.add(make_shared<sphere>(point3(4.0f, 1.0f, 0.0f), 1.0f, material3));
}



