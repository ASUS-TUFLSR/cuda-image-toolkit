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

#define BLOCK_SIZE 16
#define RADIUS 3 // blur "strength" - 3 means 7x7 window

// Shared memory tile = block size + halo on each side (RADIUS on both edges)
__global__ void blurKernel(unsigned char *input, unsigned char *output, int width, int height)
{
    __shared__ unsigned char tile[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    int tx = threadIdx.x + RADIUS;
    int ty = threadIdx.y + RADIUS;

    // Clamp to image bounds when reading (handles edges safely)
    auto clampedRead = [&](int cx, int cy) -> unsigned char
    {
        cx = min(max(cx, 0), width - 1);
        cy = min(max(cy, 0), height - 1);
        return input[cy * width + cx];
    };

    // Each thread loads its own pixel into shared memory
    tile[ty][tx] = clampedRead(x, y);

    // Threads on the block's edge also load the halo (surrounding border)
    if (threadIdx.x < RADIUS)
    {
        tile[ty][tx - RADIUS] = clampedRead(x - RADIUS, y);
        tile[ty][tx + BLOCK_SIZE] = clampedRead(x + BLOCK_SIZE, y);
    }
    if (threadIdx.y < RADIUS)
    {
        tile[ty - RADIUS][tx] = clampedRead(x, y - RADIUS);
        tile[ty + BLOCK_SIZE][tx] = clampedRead(x, y + BLOCK_SIZE);
    }

    // Corner halos (the window reads diagonals too)
    if (threadIdx.x < RADIUS && threadIdx.y < RADIUS)
    {
        tile[ty - RADIUS][tx - RADIUS] = clampedRead(x - RADIUS, y - RADIUS);
        tile[ty - RADIUS][tx + BLOCK_SIZE] = clampedRead(x + BLOCK_SIZE, y - RADIUS);
        tile[ty + BLOCK_SIZE][tx - RADIUS] = clampedRead(x - RADIUS, y + BLOCK_SIZE);
        tile[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = clampedRead(x + BLOCK_SIZE, y + BLOCK_SIZE);
    }

    __syncthreads(); // waiting until all threads fininsh loading shared memory

    if (x < width && y < height)
    {
        int sum = 0;
        int count = 0;
        for (int dy = -RADIUS; dy <= RADIUS; dy++)
        {
            for (int dx = -RADIUS; dx <= RADIUS; dx++)
            {
                sum += tile[ty + dy][tx + dx];
                count++;
            }
        }
        output[y * width + x] = (unsigned char)(sum / count);
    }
}

int main()
{
    int width, height, channels;
    unsigned char *h_input = stbi_load("images/output_gray.png", &width, &height, &channels, 1);
    if (!h_input)
    {
        printf("Failed to load image\n");
        return 1;
    }
    printf("Loaded image: %dx%d\n", width, height);

    size_t size = width * height * sizeof(unsigned char);
    unsigned char *h_output = (unsigned char *)malloc(size);

    unsigned char *d_input, *d_output;
    CHECK(cudaMalloc(&d_input, size));
    CHECK(cudaMalloc(&d_output, size));

    CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    blurKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    stbi_write_png("images/output_blur.png", width, height, 1, h_output, width);
    printf("Saved images/output_blur.png\n");

    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(h_input);
    free(h_output);

    return 0;
}