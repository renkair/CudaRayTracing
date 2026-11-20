#include <cuda_runtime.h>
#include <iostream>

#include "rtweekend.h"
#include "camera.h"
#include "hittable_list.h"
#include <curand_kernel.h>

struct hitRecordGPU {
    point3 p;
    vec3 normal;
    float t;
    bool front_face;


    __device__
    void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.
        front_face = dot(r.direction(), outward_normal) < 0.0f;
        normal = front_face ? outward_normal : -outward_normal;
    }
};



__device__ bool hit(const ray& r, interval ray_t, vec3 center, float radius, hitRecordGPU& rec);

__device__ ray get_ray(CameraData* cam, const ray& r, int i, int j, curandState* devStates, int idx);

__device__ color rayColor(SceneData * scene, ray& r, curandState* devStates, int idx);

__device__ float randomFloat(curandState* devStates, int idx, float min, float max);

__device__ vec3 sample_square(curandState* devStates, int idx);

__device__ vec3 random_unit_vector(curandState* devStates, int idx);

__device__ vec3 random_on_hemisphere(const vec3& normal, curandState* devStates, int idx);

__device__ bool scatter(MaterialGPU material, curandState* devStates, int idx, const ray& r_in, const hitRecordGPU& rec, color& attenuation, ray& scattered);


CameraData* cameraData_d;
SceneData* sceneData_d;
curandState* devStates;

__global__ void setupKernel(curandState* states, int width, int height, unsigned long seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    curand_init(seed, idx, 0, &states[idx]);
}


__global__ void renderKernel(SceneData* scene, CameraData* cam, color* sumColor, int width, int height, curandState* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < width && j < height) {
        int idx = j * width + i;

        auto offset = sample_square(states, idx);
        auto pixel_sample = cam->pixel00_loc
                + ((i + offset.x()) * cam->pixel_delta_u)
                + ((j + offset.y()) * cam->pixel_delta_v);

        auto ray_origin = cam->center;
        auto ray_direction = pixel_sample - ray_origin;
        ray r = ray(ray_origin, ray_direction);

        sumColor[idx] += rayColor(scene,r, states, idx);
    }
}
__global__ void resolveKernel(color* sumColor, unsigned char * pixels, int pass, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < width && j < height) {
        int index = (j * width + i) * 3;
        int idx = j * width + i;

        float r = linear_to_gamma(sumColor[idx].x()/pass);
        float g = linear_to_gamma(sumColor[idx].y()/pass);
        float b = linear_to_gamma(sumColor[idx].z()/pass);

        pixels[index + 0] = static_cast<unsigned char>(255.999f * r);
        pixels[index + 1] = static_cast<unsigned char>(255.999f * g);
        pixels[index + 2] = static_cast<unsigned char>(255.999f * b);
    }

}

void initDeviceData(SceneData * sceneData_h, CameraData* cameraData_h, int width, int height) {
    cudaError_t err = cudaMalloc((void **)&cameraData_d, sizeof(CameraData));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorName(err) << "\n";
        exit(1);
    }


    err = cudaMalloc((void **)&sceneData_d, sizeof(SceneData));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorName(err) << "\n";
        exit(1);
    }

    //cudaMemcpy(pixels_d, pixels_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cameraData_d, cameraData_h, sizeof(CameraData), cudaMemcpyHostToDevice);
    cudaMemcpy(sceneData_d, sceneData_h, sizeof(SceneData), cudaMemcpyHostToDevice);

    int N = 16*16*ceil(width*1.0f/16)*ceil(height*1.0f/16);

    cudaMalloc (&devStates, N * sizeof(curandState));
    dim3 block_dim(16, 16);
    dim3 grid_dim(ceil(width*1.0f/16), ceil(height*1.0f/16));
    setupKernel<<<grid_dim, block_dim>>>( devStates, width, height, 1234);
}

void renderPass(color* sumColor, unsigned char* dst, int pass, int width, int height) {
    dim3 block_dim(16, 16);
    dim3 grid_dim(ceil(width*1.0f/16), ceil(height*1.0f/16));
    renderKernel<<<grid_dim, block_dim>>>(sceneData_d, cameraData_d, sumColor, width, height, devStates);
    resolveKernel<<<grid_dim, block_dim>>>(sumColor, dst, pass, width, height);
}

void cudaDestroy() {
    cudaFree(devStates);
    cudaFree(cameraData_d);
    cudaFree(sceneData_d);
    cameraData_d = nullptr;
    sceneData_d  = nullptr;
    devStates = nullptr;


}



__device__ bool hit(const ray& r, interval ray_t, vec3 center, float radius, hitRecordGPU& rec) {
    vec3 oc = center - r.origin();
    auto a = r.direction().length_squared();
    auto h = dot(r.direction(), oc);
    auto c = oc.length_squared() - radius*radius;

    auto discriminant = h*h - a*c;
    if (discriminant < 0.0f)
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

__device__ color rayColor(SceneData * scene, ray& r, curandState* devStates, int idx) {
    int max_depth = 50;
    // vec3 unit_direction = unit_vector(r.direction());
    // auto a = 0.5*(unit_direction.y() + 1.0);
    // vec3 skyColor = (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    color attenuation(1.0f, 1.0f, 1.0f);


    for (int bounce = 0; bounce < max_depth; bounce++) {
        hitRecordGPU rec, closestRec;
        ray r_in = r;
        float closest = 1e30f;
        int closestIdx = -1;
        for (int i = 0; i < scene->sphereCount; i++) {

            float sphereRadius = scene->spheres[i].radius;
            auto sphereCenter = scene->spheres[i].center;


            if (hit(r, interval(0.001f, 1e30f), sphereCenter, sphereRadius, rec)) {
                if (rec.t < closest) {
                    closest = rec.t;
                    closestIdx = i;
                    closestRec = rec;
                }
                //cur_ray = ray(rec.p, target-rec.p)
            }
        }
        if (closestIdx < 0) {

            vec3 unit_direction = unit_vector(r.direction());
            float a = 0.5f*(unit_direction.y() + 1.0f);
            vec3 skyColor = (1.0-a)*color(1.0f, 1.0f, 1.0f) + a*color(0.5f, 0.7f, 1.0f);

            return attenuation*skyColor;
        }

        MaterialGPU material = scene->materials[closestIdx];
        ray scattered;

        bool hasNextRay = scatter(material, devStates, idx, r_in, closestRec, attenuation, scattered);
        if (!hasNextRay) {
            return color(0.0f, 0.0f, 0.0f);
        }

        if (material.type == MaterialType::Lambertian && bounce > 10) {
            float p = randomFloat(devStates, idx, 0.0f, 1.0f);
            if (p <  0.5f) {
                return color(0.0f, 0.0f, 0.0f);
            }
            attenuation *= 2.0f;
        }
        r = ray(closestRec.p, scattered.direction());
    }

    return vec3(0.0f,0.0f,0.0f); // exceeded recursion
}

__device__ float randomFloat(curandState* devStates, int idx, float min, float max)
{
    curandState localState = devStates[idx];
    float random = curand_uniform(&localState);
    devStates[idx] = localState;
    return min + random * (max - min);
}


__device__ vec3 sample_square(curandState* devStates, int idx){
    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
    return vec3(randomFloat(devStates, idx, -0.5f, 0.5f), randomFloat(devStates, idx, -0.5f, 0.5f), 0.0f);
}

__device__ vec3 random_unit_vector(curandState* devStates, int idx) {
    while (true) {
        auto p = vec3(randomFloat(devStates, idx, -1.0f, 1.0f), randomFloat(devStates, idx, -1.0f, 1.0f), randomFloat(devStates, idx, -1.0f, 1.0f));
        auto lensq = p.length_squared();
        if (FLT_MIN < lensq && lensq <= 1.0f)
            return p / sqrt(lensq);
    }
}

__device__ vec3 random_on_hemisphere(const vec3& normal, curandState* devStates, int idx) {
    vec3 on_unit_sphere = random_unit_vector(devStates, idx);
    if (dot(on_unit_sphere, normal) > 0.0f) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}



__device__ float reflectance(float cosine, float refraction_index) {
    // Use Schlick's approximation for reflectance.
    float r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*std::pow((1.0f - cosine), 5.0f);
}


__device__ bool scatter(MaterialGPU material, curandState* devStates, int idx, const ray& r_in, const hitRecordGPU& rec, color& attenuation, ray& scattered) {
    switch (material.type) {
        case MaterialType::Lambertian:
            auto scatter_direction = rec.normal + random_unit_vector(devStates, idx);
            // Catch degenerate scatter direction
            if (scatter_direction.near_zero())
                scatter_direction = rec.normal;
            scattered = ray(rec.p, scatter_direction);
            attenuation = attenuation * material.albedo;
            return true;
        case MaterialType::Metal:
            vec3 reflected = reflect(r_in.direction(), rec.normal);
            reflected = unit_vector(reflected) + (material.fuzz * random_unit_vector(devStates, idx));
            scattered = ray(rec.p, reflected);
            attenuation = attenuation * material.albedo;
            return (dot(scattered.direction(), rec.normal) > 0.0f);
        case MaterialType::Dielectric:
            attenuation = color(1.0f, 1.0f, 1.0f);
            float ri = rec.front_face ? (1.0f/material.refraction_index) : material.refraction_index;

            vec3 unit_direction = unit_vector(r_in.direction());
            float cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0f);
            float sin_theta = std::sqrt(1.0f - cos_theta*cos_theta);

            bool cannot_refract = ri * sin_theta > 1.0f;
            vec3 direction;

            if (cannot_refract || reflectance(cos_theta, ri) > randomFloat(devStates, idx, 0.0f, 1.0f))
                direction = reflect(unit_direction, rec.normal);
            else
                direction = refract(unit_direction, rec.normal, ri);

            scattered = ray(rec.p, direction);
            return true;
        case MaterialType::None:
            return false;
    }
}

