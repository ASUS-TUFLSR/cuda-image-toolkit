#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                                            \
    {                                                                                          \
        cudaError_t err = call;                                                                \
        if (err != cudaSuccess)                                                                \
        {                                                                                      \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                                           \
        }                                                                                      \
    }

__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int n = 1 << 20;
    size_t size = n * sizeof(float);

    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    for (int i = 0; i < n; i++)
    {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, size));
    CHECK(cudaMalloc(&d_b, size));
    CHECK(cudaMalloc(&d_c, size));

    CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    CHECK(cudaGetLastError());      // catches kernel launch errors
    CHECK(cudaDeviceSynchronize()); // catches kernel execution errors

    CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    bool success = true;
    int firstBad = -1;
    for (int i = 0; i < n; i++)
    {
        if (h_c[i] != 3.0f)
        {
            success = false;
            firstBad = i;
            break;
        }
    }
    if (success)
        printf("SUCCESS: all values correct\n");
    else
        printf("FAILURE at index %d, value = %f\n", firstBad, h_c[firstBad]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}