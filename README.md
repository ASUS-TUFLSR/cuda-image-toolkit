# CUDA Image Toolkit

Learning project exploring CUDA C++ for image processing, built after completing NVIDIA DLI's CUDA fundamentals course.

## Progress

- [x] Vector add (sanity check for CUDA pipeline)
- [x] Grayscale conversion
- [x] Box blur (shared memory)
- [x] Sobel edge detection
- [ ] CPU vs GPU benchmarking

## Requirements

- NVIDIA GPU + CUDA Toolkit
- MSVC (Windows) or GCC (Linux)

## Build & Run Steps

- Use x64 Native Tools Command Prompt to run CUDA commands
- nvcc src/grayscale.cu -o grayscale.exe
  grayscale.exe

## Example

Input → grayscale output using a custom CUDA kernel (2D thread indexing, one thread per pixel).
