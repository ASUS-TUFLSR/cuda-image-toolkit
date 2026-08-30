#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#define CHECK(call)                                                                            \
    {                                                                                          \
        cudaError_t err = call;                                                                \
        if (err != cudaSuccess)                                                                \
        {                                                                                      \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                                           \
        }                                                                                      \
    }

__global__ void grayscaleKernel(unsigned char *input, unsigned char *output, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        int idx = (y * width + x) * channels;
        unsigned char r = input[idx], g = input[idx + 1], b = input[idx + 2];
        output[y * width + x] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// --------- CPU version (plain sequential loop) ---------

void grayscaleCPU(unsigned char *input, unsigned char *output, int width, int height, int channels)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = (y * width + x) * channels;
            unsigned char r = input[idx], g = input[idx + 1], b = input[idx + 2];
            output[y * width + x] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }
}

int main()
{
    int width, height, channels;
    unsigned char *h_input = stbi_load("images/input.png", &width, &height, &channels, 3);
    if (!h_input)
    {
        printf("Failed to load image\n");
        return 1;
    }
    channels = 3;
    printf("IMage: %dx%d\n\n", width, height);

    size_t inputSize = width * height * channels * sizeof(unsigned char);
    size_t outputSize = width * height * sizeof(unsigned char);
    unsigned char *h_output_cpu = (unsigned char *)malloc(outputSize);
    unsigned char *h_output_gpu = (unsigned char *)malloc(outputSize);

    // --------- CPU timing ---------
    clock_t cpuStart = clock();
    grayscaleCPU(h_input, h_output_cpu, width, height, channels);
    clock_t cpuEnd = clock();
    double cpuMs = 1000.0 * (cpuEnd - cpuStart) / CLOCKS_PER_SEC;

    // --------- GPU timing (using CUDA events - more accurate than wall clock) ---------
    unsigned char *d_input, *d_output;
    CHECK(cudaMalloc(&d_input, inputSize));
    CHECK(cudaMalloc(&d_output, outputSize));
    CHECK(cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);

    // GPU warmup (not timed) — absorbs one-time CUDA init cost
    grayscaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height, channels);
    cudaDeviceSynchronize();

    // GPU timing: average over multiple runs
    int numRuns = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < numRuns; i++)
    {
        grayscaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height, channels);
    }
    cudaEventRecord(stop);

    CHECK(cudaGetLastError());
    cudaEventSynchronize(stop);

    float totalGpuMs = 0;
    cudaEventElapsedTime(&totalGpuMs, start, stop);
    float gpuMs = totalGpuMs / numRuns; // average per-run time

    CHECK(cudaMemcpy(h_output_gpu, d_output, outputSize, cudaMemcpyDeviceToHost));

    // --------- Results ---------
    printf("CPU time: %.3f ms\n", cpuMs);
    printf("GPU time: %.3f ms (kernel only, excludes memory transfer)\n", gpuMs);
    printf("Speedup: %.1fx\n\n", cpuMs / gpuMs);

    // Sanity check: outputs should match
    bool match = true;
    for (int i = 0; i < width * height; i++)
    {
        if (abs(h_output_cpu[i] - h_output_gpu[i]) > 1)
        {
            match = false;
            break;
        }
    }
    printf("Outputs match: %s\n", match ? "YESS" : "NO");

    stbi_write_png("images/output_benchmark.png", width, height, 1, h_output_gpu, width);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);

    return 0;
}