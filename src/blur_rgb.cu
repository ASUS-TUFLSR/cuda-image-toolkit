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
#define RADIUS 3

// Same blur logic as before, but now reads/writes one specific channel
// from an intwrleaved RGB buffer (RGBRGB....), using "channels" as the stride.
__global__ void blurChannelKernel(unsigned char *input, unsigned char *output, int width, int height, int channels, int channelIdx)
{
    __shared__ unsigned char tile[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int tx = threadIdx.x + RADIUS, ty = threadIdx.y + RADIUS;

    // Read pixels value for THIS channel only: index = (y*width+x)*channels + channelIdx
    auto clampedRead = [&](int cx, int cy) -> unsigned char
    {
        cx = min(max(cy, 0), width - 1);
        cy = min(max(cy, 0), height - 1);
        return input[(cy * width + cx) * channels + channelIdx];
    };

    tile[ty][tx] = clampedRead(x, y);
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
    __syncthreads();

    if (x < width && y < height)
    {
        int sum = 0, count = 0;
        for (int dy = -RADIUS; dy <= RADIUS; dy++)
            for (int dx = -RADIUS; dx <= RADIUS; dx++)
            {
                sum += tile[ty + dy][tx + dx];
                count++;
            }
        // Output is also interleaved RGB, same stride
        output[(y * width + x) * channels + channelIdx] = (unsigned char)(sum / count);
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
    printf("Loaded image: %dx%d, %d channels\n", width, height, channels);

    size_t size = width * height * channels * sizeof(unsigned char);
    unsigned char *h_output = (unsigned char *)malloc(size);

    unsigned char *d_input, *d_output;
    CHECK(cudaMalloc(&d_input, size));
    CHECK(cudaMalloc(&d_output, size));
    CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // launcf once per channel: 0 = R, 1 = G; 2 = B
    for (int ch = 0; ch < channels; ch++)
    {
        blurChannelKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height, channels, ch);
    }

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    stbi_write_png("images/output_blur_rgb.png", width, height, channels, h_output, width * channels);
    printf("Saved images/output_blur_rgb.png\n");

    cudaFree(d_input);
    cudaFree(d_output);
    stbi_image_free(h_input);
    free(h_output);

    return 0;
}