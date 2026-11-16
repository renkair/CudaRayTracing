#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

#include <memory>
#include <vector>

#include "sphere.h"

using std::make_shared;
using std::shared_ptr;

struct SphereGPU{
    double radius;
    vec3 center;
    int materialType;
};

struct SceneData {
    SphereGPU* spheres = nullptr;
    int sphereCount = 0;
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

        sceneData_h = (SceneData*)malloc(sizeof(SceneData));
        sceneData_h->spheres = spheres_d;
        sceneData_h->sphereCount = spheres_h.size();
    }
};

#endif