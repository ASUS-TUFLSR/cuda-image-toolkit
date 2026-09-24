# CUDA Image Toolkit

Learning project exploring CUDA C++ for image processing, built after completing NVIDIA DLI's CUDA fundamentals course.

## Progress

- [x] Vector add (sanity check for CUDA pipeline)
- [x] Grayscale conversion
- [x] Box blur (shared memory)
- [x] Sobel edge detection
- [x] CPU vs GPU benchmarking

## Benchmark Results
Grayscale conversion, 1054x550 image, averaged over 100 runs:

| | Time | 
|---|---|
| CPU (single-threaded loop) | 9.000 ms |
| GPU (CUDA kernel only) | 0.038 ms |
| **Speedup** | **235.8x** |

*Note: GPU timing excludes host-device memory transfer to isolate compute performance. First-kernel-launch overhead (~20-30ms CUDA context init) is absorbed via a warmup call before timing.*

## Example Output
| Original | Grayscale | Blur | Edge Detection |
|---|---|---|---|
| ![input](images/input.png) | ![gray](images/output_gray.png) | ![blur](images/output_blur.png) | ![edges](images/output_edges.png) |

## Requirements

- NVIDIA GPU + CUDA Toolkit
- MSVC (Windows) or GCC (Linux)

## Build & Run Steps

- Use x64 Native Tools Command Prompt to run CUDA commands
- nvcc src/vector_add.cu -o vector_add.exe
  vector_add.exe

- nvcc src/grayscale.cu -o grayscale.exe
  grayscale.exe

- nvcc src/blur.cu -o blur.exe
  blur.exe
  
- nvcc src/edge_detect.cu -o edge_detect.exe
  edge_detect.exe

- nvcc src/benchmark.cu -o benchmark.exe
  benchmark.exe

## Example

Input → grayscale output using a custom CUDA kernel (2D thread indexing, one thread per pixel).

## Now Extending the benchmark to blur/edge detection to heavier kernels
- Created benchmark_all.cu this will add CPU versions of blur/edge detection then 3x all three CPUvsGPU
- Run these commands:  
- nvcc src/benchmark_all.cu -o benchmark_all.exe 
- benchmark_all.exe

## Results of above test
Image: 1054x550
Kernel               |    CPU (ms)    |  GPU (ms)  |  Speedup
Grayscale            |       9.000    |    0.032   |  285.2x
Box Blur (7x7)       |     426.000    |    0.098   |  4331.4x
Sobel Edge           |      74.000    |    0.042   |  1751.9x

- Now here blur jumped so much higher than grayscale: each pixel now does 49 reads + sum instead of grayscale's 3 reads + weighted sum — CPU cost scales up hard (426ms!) since it's doing that sequentially, one pixel at a time. GPU barely notices because thousands of cores absorb that extra per-pixel work in parallel..