#include <cuda_runtime.h>

struct MaterialsGPU {
    int type = -1; // 0 - lambertian, 1 - metal,  2 - dielectric
    float albedo;
    float fuzz;
    float refraction_index;
};