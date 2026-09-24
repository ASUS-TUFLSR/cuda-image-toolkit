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

#define BLOCK_SIZE 16
#define BLUR_RADIUS 3
#define SOBEL_RADIUS 1

// ---------------- GPU Kernels ----------------
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

__global__ void blurKernel(unsigned char *input, unsigned char *output, int width, int height)
{
    __shared__ unsigned char tile[BLOCK_SIZE + 2 * BLUR_RADIUS][BLOCK_SIZE + 2 * BLUR_RADIUS];
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int tx = threadIdx.x + BLUR_RADIUS, ty = threadIdx.y + BLUR_RADIUS;

    auto clampedRead = [&](int cx, int cy) -> unsigned char
    {
        cx = min(max(cx, 0), width - 1);
        cy = min(max(cy, 0), height - 1);
        return input[cy * width + cx];
    };

    tile[ty][tx] = clampedRead(x, y);
    if (threadIdx.x < BLUR_RADIUS)
    {
        tile[ty][tx - BLUR_RADIUS] = clampedRead(x - BLUR_RADIUS, y);
        tile[ty][tx + BLOCK_SIZE] = clampedRead(x + BLOCK_SIZE, y);
    }
    if (threadIdx.y < BLUR_RADIUS)
    {
        tile[ty - BLUR_RADIUS][tx] = clampedRead(x, y - BLUR_RADIUS);
        tile[ty + BLOCK_SIZE][tx] = clampedRead(x, y + BLOCK_SIZE);
    }
    __syncthreads();

    if (x < width && y < height)
    {
        int sum = 0, count = 0;
        for (int dy = -BLUR_RADIUS; dy <= BLUR_RADIUS; dy++)
            for (int dx = -BLUR_RADIUS; dx <= BLUR_RADIUS; dx++)
            {
                sum += tile[ty + dy][tx + dx];
                count++;
            }
        output[y * width + x] = (unsigned char)(sum / count);
    }
}

__global__ void sobelKernel(unsigned char *input, unsigned char *output, int width, int height)
{
    __shared__ unsigned char tile[BLOCK_SIZE + 2 * SOBEL_RADIUS][BLOCK_SIZE + 2 * SOBEL_RADIUS];
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int tx = threadIdx.x + SOBEL_RADIUS, ty = threadIdx.y + SOBEL_RADIUS;

    auto clampedRead = [&](int cx, int cy) -> unsigned char
    {
        cx = min(max(cx, 0), width - 1);
        cy = min(max(cy, 0), height - 1);
        return input[cy * width + cx];
    };

    tile[ty][tx] = clampedRead(x, y);
    if (threadIdx.x < SOBEL_RADIUS)
    {
        tile[ty][tx - SOBEL_RADIUS] = clampedRead(x - SOBEL_RADIUS, y);
        tile[ty][tx + BLOCK_SIZE] = clampedRead(x + BLOCK_SIZE, y);
    }
    if (threadIdx.y < SOBEL_RADIUS)
    {
        tile[ty - SOBEL_RADIUS][tx] = clampedRead(x, y - SOBEL_RADIUS);
        tile[ty + BLOCK_SIZE][tx] = clampedRead(x, y + BLOCK_SIZE);
    }
    if (threadIdx.x < SOBEL_RADIUS && threadIdx.y < SOBEL_RADIUS)
    {
        tile[ty - SOBEL_RADIUS][tx - SOBEL_RADIUS] = clampedRead(x - SOBEL_RADIUS, y - SOBEL_RADIUS);
        tile[ty - SOBEL_RADIUS][tx + BLOCK_SIZE] = clampedRead(x + BLOCK_SIZE, y - SOBEL_RADIUS);
        tile[ty + BLOCK_SIZE][tx - SOBEL_RADIUS] = clampedRead(x - SOBEL_RADIUS, y + BLOCK_SIZE);
        tile[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = clampedRead(x + BLOCK_SIZE, y + BLOCK_SIZE);
    }
    __syncthreads();

    if (x < width && y < height)
    {
        int Gx = -tile[ty - 1][tx - 1] - 2 * tile[ty][tx - 1] - tile[ty + 1][tx - 1] + tile[ty - 1][tx + 1] + 2 * tile[ty][tx + 1] + tile[ty + 1][tx + 1];
        int Gy = -tile[ty - 1][tx - 1] - 2 * tile[ty - 1][tx] - tile[ty - 1][tx + 1] + tile[ty + 1][tx - 1] + 2 * tile[ty + 1][tx] + tile[ty + 1][tx + 1];
        int mag = (int)sqrtf((float)(Gx * Gx + Gy * Gy));
        output[y * width + x] = (unsigned char)min(max(mag, 0), 255);
    }
}

// ---------------- CPU versions ---------------
void grayscaleCPU(unsigned char *input, unsigned char *output, int width, int height, int channels)
{
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int idx = (y * width + x) * channels;
            output[y * width + x] = (unsigned char)(0.299f * input[idx] + 0.587f * input[idx + 1] + 0.114f * input[idx + 2]);
        }
}

void blurCPU(unsigned char *input, unsigned char *output, int width, int height)
{
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int sum = 0, count = 0;
            for (int dy = -BLUR_RADIUS; dy <= BLUR_RADIUS; dy++)
                for (int dx = -BLUR_RADIUS; dx <= BLUR_RADIUS; dx++)
                {
                    int cx = min(max(x + dx, 0), width - 1);
                    int cy = min(max(y + dy, 0), height - 1);
                    sum += input[cy * width + cx];
                    count++;
                }
            output[y * width + x] = (unsigned char)(sum / count);
        }
}

void sobelCPU(unsigned char *input, unsigned char *output, int width, int height)
{
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            auto px = [&](int cx, int cy)
            {
                cx = min(max(cx, 0), width - 1);
                cy = min(max(cy, 0), height - 1);
                return input[cy * width + cx];
            };
            int Gx = -px(x - 1, y - 1) - 2 * px(x - 1, y) - px(x - 1, y + 1) + px(x + 1, y - 1) + 2 * px(x + 1, y) + px(x + 1, y + 1);
            int Gy = -px(x - 1, y - 1) - 2 * px(x, y - 1) - px(x + 1, y - 1) + px(x - 1, y + 1) + 2 * px(x, y + 1) + px(x + 1, y + 1);
            int mag = (int)sqrtf((float)(Gx * Gx + Gy * Gy));
            output[y * width + x] = (unsigned char)min(max(mag, 0), 255);
        }
}

// ---------------- Timig helper ----------------
float timeGPUKernel(void (*launch)(unsigned char *, unsigned char *, int, int), unsigned char *d_in, unsigned char *d_out, int width, int height, int numRuns)
{
    launch(d_in, d_out, width, height); // warmup
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < numRuns; i++)
        launch(d_in, d_out, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float totalMs = 0;
    cudaEventElapsedTime(&totalMs, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return totalMs / numRuns;
}

// Wrapper functions matching the function pointer signature (fixed block/grid config)
dim3 g_blocksPerGrid, g_threadsPerBlock;
void launchBlur(unsigned char *in, unsigned char *out, int w, int h)
{
    blurKernel<<<g_blocksPerGrid, g_threadsPerBlock>>>(in, out, w, h);
}
void launchSobel(unsigned char *in, unsigned char *out, int w, int h)
{
    sobelKernel<<<g_blocksPerGrid, g_threadsPerBlock>>>(in, out, w, h);
}

int main()
{
    int width, height, channels;
    unsigned char *h_rgb = stbi_load("images/input.png", &width, &height, &channels, 3);
    if (!h_rgb)
    {
        printf("Failed to load image\n");
        return 1;
    }
    channels = 3;
    printf("Image: %dx%d\n\n", width, height);

    size_t rgbSize = width * height * channels;
    size_t graySize = width * height;

    unsigned char *h_gray_cpu = (unsigned char *)malloc(graySize);
    unsigned char *h_gray_gpu = (unsigned char *)malloc(graySize);
    unsigned char *h_blur_cpu = (unsigned char *)malloc(graySize);
    unsigned char *h_blur_gpu = (unsigned char *)malloc(graySize);
    unsigned char *h_edge_cpu = (unsigned char *)malloc(graySize);
    unsigned char *h_edge_gpu = (unsigned char *)malloc(graySize);

    int numRuns = 100;
    g_threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    g_blocksPerGrid = dim3((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    unsigned char *d_rgb, *d_gray, *d_blur, *d_edge;
    CHECK(cudaMalloc(&d_rgb, rgbSize));
    CHECK(cudaMalloc(&d_gray, graySize));
    CHECK(cudaMalloc(&d_blur, graySize));
    CHECK(cudaMalloc(&d_edge, graySize));
    CHECK(cudaMemcpy(d_rgb, h_rgb, rgbSize, cudaMemcpyHostToDevice));

    // ---- Grayscale -----
    clock_t c1 = clock();
    grayscaleCPU(h_rgb, h_gray_cpu, width, height, channels);
    double grayCpuMs = 1000.0 * (clock() - c1) / CLOCKS_PER_SEC;

    grayscaleKernel<<<g_blocksPerGrid, g_threadsPerBlock>>>(d_rgb, d_gray, width, height, channels); // warmup
    cudaDeviceSynchronize();
    cudaEvent_t s1, e1;
    cudaEventCreate(&s1);
    cudaEventCreate(&e1);
    cudaEventRecord(s1);
    for (int i = 0; i < numRuns; i++)
        grayscaleKernel<<<g_blocksPerGrid, g_threadsPerBlock>>>(d_rgb, d_gray, width, height, channels);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float grayGpuTotal;
    cudaEventElapsedTime(&grayGpuTotal, s1, e1);
    float grayGpuMs = grayGpuTotal / numRuns;
    CHECK(cudaMemcpy(h_gray_gpu, d_gray, graySize, cudaMemcpyDeviceToHost));

    // ---- Blur (runs on grayscale output) ----
    CHECK(cudaMemcpy(d_gray, h_gray_cpu, graySize, cudaMemcpyHostToDevice)); // ensure same input for fair timing

    clock_t c2 = clock();
    blurCPU(h_gray_cpu, h_blur_cpu, width, height);
    double blurCpuMs = 1000.0 * (clock() - c2) / CLOCKS_PER_SEC;

    float blurGpuMs = timeGPUKernel(launchBlur, d_gray, d_blur, width, height, numRuns);
    CHECK(cudaMemcpy(h_blur_gpu, d_blur, graySize, cudaMemcpyDeviceToHost));

    // ---- Sobel edge detection (run on grayscale output) ----
    clock_t c3 = clock();
    sobelCPU(h_gray_cpu, h_edge_cpu, width, height);
    double edgeCpuMs = 1000.0 * (clock() - c3) / CLOCKS_PER_SEC;

    float edgeGpuMs = timeGPUKernel(launchSobel, d_gray, d_edge, width, height, numRuns);
    CHECK(cudaMemcpy(h_edge_gpu, d_edge, graySize, cudaMemcpyDeviceToHost));

    // ---- Results ----
    printf("%-20s %12s %12s %10s\n", "Kernel", "CPU (ms)", "GPU (ms)", "Speedup");
    printf("%-20s %12.3f %12.3f %9.1fx\n", "Grayscale", grayCpuMs, grayGpuMs, grayCpuMs / grayGpuMs);
    printf("%-20s %12.3f %12.3f %9.1fx\n", "Box Blur (7x7)", blurCpuMs, blurGpuMs, blurCpuMs / blurGpuMs);
    printf("%-20s %12.3f %12.3f %9.1fx\n", "Sobel Edge", edgeCpuMs, edgeGpuMs, edgeCpuMs / edgeGpuMs);

    stbi_write_png("images/bench_gray.png", width, height, 1, h_gray_gpu, width);
    stbi_write_png("images/bench_blur.png", width, height, 1, h_blur_gpu, width);
    stbi_write_png("images/bench_edge.png", width, height, 1, h_edge_gpu, width);

    cudaFree(d_rgb);
    cudaFree(d_gray);
    cudaFree(d_blur);
    cudaFree(d_edge);
    stbi_image_free(h_rgb);
    free(h_gray_cpu);
    free(h_gray_gpu);
    free(h_blur_cpu);
    free(h_blur_gpu);
    free(h_edge_cpu);
    free(h_edge_gpu);

    return 0;
}