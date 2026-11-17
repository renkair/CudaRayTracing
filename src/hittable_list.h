#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

#include <memory>
#include <vector>

#include "sphere.h"
#include "material.h"

using std::make_shared;
using std::shared_ptr;

struct SphereGPU{
    double radius;
    vec3 center;
};

struct MaterialGPU {
    int type = -1; // 0 - lambertian, 1 - metal,  2 - dielectric
    color albedo;
    double fuzz;
    double refraction_index;
};

struct SceneData {
    SphereGPU* spheres = nullptr;
    int sphereCount = 0;
    MaterialGPU* materials = nullptr;
};

class hittable_list : public hittable {
public:
    std::vector<shared_ptr<hittable>> objects;
    SceneData* sceneData_h = nullptr;

    hittable_list() {}
    hittable_list(shared_ptr<hittable> object) { add(object); }

    void clear() { objects.clear(); }

    void add(shared_ptr<hittable> object) {
        objects.push_back(object);
    }

    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.max;

        for (const auto& object : objects) {
            if (object->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }

    void prepareSceneData() {
        // Prepare Sphere Data
        std::vector<SphereGPU> spheres_h;
        spheres_h.reserve(objects.size()); // temp
        for (auto& object : objects) {
            const auto sphere_ptr = std::dynamic_pointer_cast<sphere>(object);
            if (sphere_ptr != nullptr) {
                spheres_h.push_back(SphereGPU{sphere_ptr->getRadius(), sphere_ptr->getCenter()});
            }
        }
        SphereGPU* spheres_d = nullptr;
        cudaError_t err = cudaMalloc((void**)&spheres_d, sizeof(SphereGPU) * spheres_h.size());
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorName(err) << "\n";
            exit(1);
        }
        cudaMemcpy(spheres_d, spheres_h.data(), sizeof(SphereGPU) *  spheres_h.size(), cudaMemcpyHostToDevice);


        // Prepare Materials Data
        std::vector<MaterialGPU> materials_h;
        materials_h.reserve(objects.size());
        for (auto& object : objects) {
            const auto sphere_ptr = std::dynamic_pointer_cast<sphere>(object);
            auto material_ptr = std::dynamic_pointer_cast<material>(sphere_ptr->getMat());
            if (auto lambertian_ptr = std::dynamic_pointer_cast<lambertian>(material_ptr)) {
                // Material is Lambertian
                materials_h.push_back(MaterialGPU{0, lambertian_ptr->getAlbedo(), -1, -1});
            }else if (auto metal_ptr = std::dynamic_pointer_cast<metal>(material_ptr)) {
                // Material is Metal
                materials_h.push_back(MaterialGPU{1, metal_ptr->getAlbedo(), metal_ptr->getFuzz(), -1});

            }else if (auto dielectric_ptr = std::dynamic_pointer_cast<dielectric>(material_ptr)) {
                // Material is dielectric
                materials_h.push_back(MaterialGPU{2, color(), -1, dielectric_ptr->getRefractionIndex()});
            }
        }
        MaterialGPU* materials_d = nullptr;
        err = cudaMalloc((void**)&materials_d, sizeof(MaterialGPU)*materials_h.size());
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorName(err) << "\n";
        }
        cudaMemcpy(materials_d, materials_h.data(), sizeof(MaterialGPU) * materials_h.size(), cudaMemcpyHostToDevice);


        sceneData_h = (SceneData*)malloc(sizeof(SceneData));
        sceneData_h->spheres = spheres_d;
        sceneData_h->materials = materials_d;
        sceneData_h->sphereCount = spheres_h.size();

    }
};

#endif