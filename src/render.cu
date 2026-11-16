#include <cuda_runtime.h>
#include <iostream>

#include "rtweekend.h"
#include "camera.h"
#include "hittable_list.h"

__device__
bool hit(const ray& r, interval ray_t, vec3 center, float radius);


__global__
void renderKernel(SceneData* scene, CameraData* cam, unsigned char * pixels, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < width && j < height) {
        int index = (j * width + i) * 3;
        color pixel_color(0,0,0);
        //ray r = get_ray(i, j);

        //auto offset = sample_square();
        auto pixel_sample = cam->pixel00_loc
                          + (i * cam->pixel_delta_u)
                          + (j * cam->pixel_delta_v);

        auto ray_origin = cam->center;
        auto ray_direction = pixel_sample - ray_origin;

        ray r = ray(ray_origin, ray_direction);

        //pixel_color += ray_color(r, max_depth, world);
        //printf("scene spheres count: %d\n", scene->sphereCount);
        for (int i = 0; i < scene->sphereCount; i++) {
            auto sphereRadius = scene->spheres[i].radius;
            auto sphereCenter = scene->spheres[i].center;
            if (hit(r, interval(0.001, 1e30), sphereCenter, sphereRadius)) {
                pixel_color = color(0.1f, 0.1f, 0.5f);
            }

        }



        pixels[index + 0] = static_cast<unsigned char>(255.999 * pixel_color.x());
        pixels[index + 1] = static_cast<unsigned char>(255.999 * pixel_color.y());
        pixels[index + 2] = static_cast<unsigned char>(255.999 * pixel_color.z());
    }
}

void render_d(SceneData * sceneData_h, CameraData* cameraData_h, unsigned char *pixels_h, int width, int height) {
    unsigned char* pixels_d;
    int size = width * height * 3 * sizeof(unsigned char);

    cudaError_t err = cudaMalloc((void **)&pixels_d, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorName(err) << "\n";
        exit(1);
    }

    CameraData* cameraData_d;
    err = cudaMalloc((void **)&cameraData_d, sizeof(CameraData));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorName(err) << "\n";
        exit(1);
    }

    SceneData* sceneData_d;
    err = cudaMalloc((void **)&sceneData_d, sizeof(SceneData));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorName(err) << "\n";
        exit(1);
    }

    cudaMemcpy(pixels_d, pixels_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cameraData_d, cameraData_h, sizeof(CameraData), cudaMemcpyHostToDevice);
    cudaMemcpy(sceneData_d, sceneData_h, sizeof(SceneData), cudaMemcpyHostToDevice);

    dim3 block_dim(32, 32);
    dim3 grid_dim(ceil(width*1.0f/32), ceil(height*1.0f/32));
    renderKernel<<<grid_dim, block_dim>>>(sceneData_d, cameraData_d, pixels_d, width, height);

    cudaMemcpy(pixels_h, pixels_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(cameraData_h, cameraData_d, sizeof(CameraData), cudaMemcpyDeviceToHost);
    cudaMemcpy(sceneData_h, sceneData_d, sizeof(SceneData), cudaMemcpyDeviceToHost);
    // update camera data in host

    cudaFree(pixels_d);
    cudaFree(cameraData_d);
    cudaFree(sceneData_d);
}


__device__
bool hit(const ray& r, interval ray_t, vec3 center, float radius) {
    vec3 oc = center - r.origin();
    auto a = r.direction().length_squared();
    auto h = dot(r.direction(), oc);
    auto c = oc.length_squared() - radius*radius;

    auto discriminant = h*h - a*c;
    if (discriminant < 0)
        return false;

    auto sqrtd = std::sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (h - sqrtd) / a;
    if (!ray_t.surrounds(root)) {
        root = (h + sqrtd) / a;
        if (!ray_t.surrounds(root))
            return false;
    }

    // rec.t = root;
    // rec.p = r.at(rec.t);
    // vec3 outward_normal = (rec.p - center) / radius;
    // rec.set_face_normal(r, outward_normal);
    // rec.mat = mat;

    return true;
}

