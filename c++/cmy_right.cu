// GPU parallel (right): the correct way to do it.
// One host->device transfer, one vectorized kernel launch, one device->host
// transfer -- mirrors python/cmy.py's gpu_parallel_right() and
// julia/cmy.jl's gpu_parallel_right().
#include <cstdio>
#include <cstdlib>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "vendor/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vendor/stb_image_write.h"

constexpr bool WRITE_FILE = true;
constexpr const char* FILE_IN = "../big.jpg";
constexpr const char* FILE_OUT = "../big_out_cpp.png";
constexpr int CHANNELS = 3;

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

__global__ void rgb_to_cmy_kernel(const unsigned char* in, unsigned char* out,
                                   size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 255 - in[i];
    }
}

static void warmup_gpu() {
    unsigned char *d_dummy;
    CUDA_CHECK(cudaMalloc(&d_dummy, 2));
    rgb_to_cmy_kernel<<<1, 2>>>(d_dummy, d_dummy, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_dummy));
}

static double gpu_parallel_right(const unsigned char* h_in, unsigned char* h_out,
                                  size_t n) {
    auto start = std::chrono::high_resolution_clock::now();

    unsigned char *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, n));
    CUDA_CHECK(cudaMalloc(&d_out, n));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, n, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    rgb_to_cmy_kernel<<<blocks, threads>>>(d_in, d_out, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, n, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

int main() {
    int width, height, channels;
    unsigned char* img = stbi_load(FILE_IN, &width, &height, &channels, CHANNELS);
    if (!img) {
        fprintf(stderr, "Failed to load %s: %s\n", FILE_IN, stbi_failure_reason());
        return 1;
    }
    size_t n = (size_t)width * height * CHANNELS;
    printf("Loaded %s: %dx%d\n", FILE_IN, width, height);

    unsigned char* out = (unsigned char*)malloc(n);

    warmup_gpu(); // one-time CUDA context/kernel JIT cost, excluded from timing

    double elapsed = gpu_parallel_right(img, out, n);
    printf("GPU parallel (right): %.4fs\n", elapsed);

    if (WRITE_FILE) {
        stbi_write_png(FILE_OUT, width, height, CHANNELS, out, width * CHANNELS);
    }

    stbi_image_free(img);
    free(out);
    return 0;
}
