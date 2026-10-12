#include <stdio.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#define CHECK(call)                                                                           \
    {                                                                                         \
        cudaError_t err = call;                                                               \
        if (err != cudaSuccess)                                                               \
        {                                                                                     \
            printf("CUDA error at %s:%d -%s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                                          \
        }                                                                                     \
    }

#define BLOCK_SIZE 16
#define RADIUS 1 // Sobel uses a 3x3 window

__global__ void sobelChannelKernel(unsigned char *input, unsigned char *output, int width, int height, int channels, int channelIdx)
{
    __shared__ unsigned char title[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    int tx = threadIdx.x + RADIUS;
    int ty = threadIdx.y + RADIUS;

    auto clampedRead = [&](int cx, int cy) -> unsigned char
    {
        cx = min(max(cx, 0), width - 1);
        cy = min(max(cy, 0), height - 1);
        return input[(cy * width + cx) * channels + channelIdx];
    };

    title[ty][tx] = clampedRead(x, y);

    if (threadIdx.x < RADIUS)
    {
        title[ty][tx - RADIUS] = clampedRead(x - RADIUS, y);
        title[ty][tx + BLOCK_SIZE] = clampedRead(x + BLOCK_SIZE, y);
    }
    if (threadIdx.y < RADIUS)
    {
        title[ty - RADIUS][tx] = clampedRead(x, y - RADIUS);
        title[ty + BLOCK_SIZE][tx] = clampedRead(x, y + BLOCK_SIZE);
    }
    // Corners of the halo (needed for a 3x3 window, wasn't needed for seperable blur)
    if (threadIdx.x < RADIUS && threadIdx.y < RADIUS)
    {
        title[ty - RADIUS][tx - RADIUS] = clampedRead(x - RADIUS, y - RADIUS);
        title[ty - RADIUS][tx + BLOCK_SIZE] = clampedRead(x + BLOCK_SIZE, y - RADIUS);
        title[ty + BLOCK_SIZE][tx - RADIUS] = clampedRead(x - RADIUS, y + BLOCK_SIZE);
        title[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = clampedRead(x + BLOCK_SIZE, y + BLOCK_SIZE);
    }

    __syncthreads();

    if (x < width && y < height)
    {
        // Horizontal gradient (Gx) and vertical gradient (Gy) kernels
        int Gx = -title[ty - 1][tx - 1] - 2 * title[ty][tx - 1] - title[ty + 1][tx - 1] + title[ty - 1][tx + 1] + 2 * title[ty][tx + 1] + title[ty + 1][tx + 1];

        int Gy = -title[ty - 1][tx - 1] - 2 * title[ty - 1][tx] - title[ty - 1][tx + 1] + title[ty + 1][tx - 1] + 2 * title[ty + 1][tx] + title[ty + 1][tx + 1];

        int magnitude = (int)sqrtf((float)(Gx * Gx + Gy * Gy));
        magnitude = min(max(magnitude, 0), 255);

        output[(y * width + x) * channels + channelIdx] = (unsigned char)magnitude;
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
    printf("Loaded image: %dx%d\n", width, height);

        size_t size = width * height * channels * sizeof(unsigned char);
    unsigned char *h_output = (unsigned char *)malloc(size);

    unsigned char *d_input, *d_output;
    CHECK(cudaMalloc(&d_input, size));
    CHECK(cudaMalloc(&d_output, size));

    CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int ch = 0; ch < channels; ch++)
    {
        sobelChannelKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height, channels, ch);
    }

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    stbi_write_png("images/output_edges_rgb.png", width, height, channels, h_output, width * channels);
    printf("Saved images/output_edges_rgb.png\n");

    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(h_input);
    free(h_output);

    return 0;
}