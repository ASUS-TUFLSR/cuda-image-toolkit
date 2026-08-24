#include <stdio.h>
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

// One thread per pixel. Each thread reads R,G,B and writes 1 gray value.
__global__ void grayscaleKernel(unsigned char *input, unsigned char *output,
                                int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = (y * width + x) * channels;
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        output[y * width + x] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
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
    channels = 3; // we forced 3 channels (RGB) above
    printf("Loaded image: %dx%d, %d channels\n", width, height, channels);

    size_t inputSize = width * height * channels * sizeof(unsigned char);
    size_t outputSize = width * height * sizeof(unsigned char);

    unsigned char *h_output = (unsigned char *)malloc(outputSize);

    unsigned char *d_input, *d_output;
    CHECK(cudaMalloc(&d_input, inputSize));
    CHECK(cudaMalloc(&d_output, outputSize));

    CHECK(cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);

    grayscaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height, channels);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost));

    stbi_write_png("images/output_gray.png", width, height, 1, h_output, width);
    printf("Saved images/output_gray.png\n");

    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(h_input);
    free(h_output);

    return 0;
}