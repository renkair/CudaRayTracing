#include <cuda_runtime.h>
#include <iostream>

__global__
void renderKernel(unsigned char * pixels, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < width && j < height) {
        int index = (j * width + i) * 3;
        color pixel_color(0,0,0);
        ray r = get_ray(i, j);
        pixel_color += ray_color(r, max_depth, world);
        pixels[index + 0] = static_cast<unsigned char>(255.999 * i*1.0f/width);
        pixels[index + 1] = static_cast<unsigned char>(255.999 * j*1.0f/height);
        pixels[index + 2] = static_cast<unsigned char>(255.999 * 0.5f);
    }
}

void render_d(unsigned char *pixels_h, int width, int height) {
    unsigned char* pixels_d;
    int size = width * height * 3 * sizeof(unsigned char);

    cudaError_t err = cudaMalloc((void **)&pixels_d, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorName(err) << "\n";
        exit(1);
    }

    cudaMemcpy(pixels_d, pixels_h, size, cudaMemcpyHostToDevice);

    dim3 block_dim(32, 32);
    dim3 grid_dim(ceil(width*1.0f/32), ceil(height*1.0f/32));
    renderKernel<<<grid_dim, block_dim>>>(pixels_d, width, height);

    cudaMemcpy(pixels_h, pixels_d, size, cudaMemcpyDeviceToHost);

    cudaFree(pixels_d);
}

