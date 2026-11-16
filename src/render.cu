#include <cuda_runtime.h>
#include <iostream>

#include "rtweekend.h"
#include "camera.h"
#include "hittable_list.h"
#include <curand_kernel.h>

struct hitRecordGPU {
    point3 p;
    vec3 normal;
    double t;
    bool front_face;

    __device__
    void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};



__device__ bool hit(const ray& r, interval ray_t, vec3 center, float radius, hitRecordGPU& rec);

__device__ ray get_ray(CameraData* cam, const ray& r, int i, int j, curandState* devStates, int idx);

__device__ color rayColor(SceneData * scene, ray& r, curandState* globalState, int idx);

__device__ float randomFloat(curandState* globalState, int idx, float min, float max);

__device__ vec3 sample_square(curandState* devStates, int idx);

__device__ vec3 random_unit_vector(curandState* globalState, int idx);

__device__ vec3 random_on_hemisphere(const vec3& normal, curandState* globalState, int idx);



__global__ void setupKernel( curandState * states, int width, int height, unsigned long seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    curand_init(seed, idx, 0, &states[idx]);
}



__global__
void renderKernel(SceneData* scene, CameraData* cam, unsigned char * pixels, int width, int height, curandState* devStates) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < width && j < height) {
        int index = (j * width + i) * 3;

        //ray r = get_ray(i, j);

        int idx = j * width + i;
        
        color pixel_color = color(0.0f, 0.0f, 0.0f);
        int samples = 16;
        for (int k = 0; k < samples; k++) {
            auto offset = sample_square(devStates, idx);
            auto pixel_sample = cam->pixel00_loc
                              + ((i + offset.x()) * cam->pixel_delta_u)
                              + ((j + offset.y()) * cam->pixel_delta_v);

            auto ray_origin = cam->center;
            auto ray_direction = pixel_sample - ray_origin;
            ray r = ray(ray_origin, ray_direction);

            pixel_color += rayColor(scene,r, devStates, idx);
        }
        pixel_color *= 1.0 / samples;

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

    dim3 block_dim(16, 16);
    dim3 grid_dim(ceil(width*1.0f/16), ceil(height*1.0f/16));

    int N = 16*16*ceil(width*1.0f/16)*ceil(height*1.0f/16);
    curandState* devStates;
    cudaMalloc (&devStates, N * sizeof(curandState));
    setupKernel<<<grid_dim, block_dim>>>(devStates, width, height, 1234);
    renderKernel<<<grid_dim, block_dim>>>(sceneData_d, cameraData_d, pixels_d, width, height, devStates);

    cudaMemcpy(pixels_h, pixels_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(cameraData_h, cameraData_d, sizeof(CameraData), cudaMemcpyDeviceToHost);
    cudaMemcpy(sceneData_h, sceneData_d, sizeof(SceneData), cudaMemcpyDeviceToHost);
    // update camera data in host

    cudaFree(pixels_d);
    cudaFree(cameraData_d);
    cudaFree(sceneData_d);
}




__device__ bool hit(const ray& r, interval ray_t, vec3 center, float radius, hitRecordGPU& rec) {
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


    rec.t = root;
    rec.p = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    //rec.mat = mat;

    return true;
}


__device__ ray get_ray(CameraData* cam, const ray& r, int i, int j, curandState* devStates, int idx){
    // Construct a camera ray originating from the defocus disk and directed at a randomly
    // sampled point around the pixel location i, j.

    auto offset = sample_square(devStates, idx);
    auto pixel_sample = cam->pixel00_loc
                      + ((i + offset.x()) * cam->pixel_delta_u)
                      + ((j + offset.y()) * cam->pixel_delta_v);

    //auto ray_origin = (cam->defocus_angle <= 0) ? cam->center : cam->defocus_disk_sample();
    auto ray_origin = cam->center;
    auto ray_direction = pixel_sample - ray_origin;

    return ray(ray_origin, ray_direction);
}

__device__ color rayColor(SceneData * scene, ray& r, curandState* state, int idx) {
    int max_depth = 50;
    // vec3 unit_direction = unit_vector(r.direction());
    // auto a = 0.5*(unit_direction.y() + 1.0);
    // vec3 skyColor = (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    color attenuation(1.0, 1.0, 1.0);

    for (int bounce = 0; bounce < max_depth; bounce++) {
        hitRecordGPU rec;
        double closest = 1e30;
        int cloestIdx = -1;
        for (int i = 0; i < scene->sphereCount; i++) {

            auto sphereRadius = scene->spheres[i].radius;
            auto sphereCenter = scene->spheres[i].center;

            if (hit(r, interval(0.001, 1e30), sphereCenter, sphereRadius, rec)) {
                if (rec.t < closest) {
                    closest = rec.t;
                    cloestIdx = i;
                }
                //cur_ray = ray(rec.p, target-rec.p)
            }
        }
        if (cloestIdx < 0) {
            vec3 unit_direction = unit_vector(r.direction());
            auto a = 0.5*(unit_direction.y() + 1.0);
            vec3 skyColor = (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
            return attenuation*skyColor;
        }


        hit(r, interval(0.001, 1e30), scene->spheres[cloestIdx].center, scene->spheres[cloestIdx].radius, rec);
        attenuation *= 0.5f;
        vec3 direction = random_on_hemisphere(rec.normal, state, idx);
        r = ray(rec.p, direction);
    }

    return vec3(0.0,0.0,0.0); // exceeded recursion
}

__device__ float randomFloat(curandState* globalState, int idx, float min, float max)
{
    curandState localState = globalState[idx];
    float random = curand_uniform(&localState);
    globalState[idx] = localState;
    return min + random * (max - min);
}


__device__ vec3 sample_square(curandState* devStates, int idx){
    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
    return vec3(randomFloat(devStates, idx, -0.5, 0.5), randomFloat(devStates, idx, -0.5, 0.5), 0);
}

__device__ vec3 random_unit_vector(curandState* globalState, int idx) {
    while (true) {
        auto p = vec3(randomFloat(globalState, idx, -1.0f, 1.0f), randomFloat(globalState, idx, -1.0f, 1.0f), randomFloat(globalState, idx, -1.0f, 1.0f));
        auto lensq = p.length_squared();
        if (1e-160 < lensq && lensq <= 1)
            return p / sqrt(lensq);
    }
}

__device__ vec3 random_on_hemisphere(const vec3& normal, curandState* globalState, int idx) {
    vec3 on_unit_sphere = random_unit_vector(globalState, idx);
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

