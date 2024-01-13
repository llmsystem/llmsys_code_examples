#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <fstream>


#define RADIUS 2
#define THREADS_PER_BLOCK 4


__global__ void WindowSumSimpleKernel(float* A, float *B, int n) {
    int out_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (out_idx < n) {
        float sum = 0;
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
            sum += A[dx + out_idx + RADIUS];
        }
        B[out_idx] = sum;
    }
}

__global__ void WindowSumSharedKernel(float* A, float *B, int size_a, int size_b) {
    __shared__ float temp[THREADS_PER_BLOCK + 2 * RADIUS];
    int base = blockDim.x * blockIdx.x;
    int out_idx = base + threadIdx.x;
    if (base + threadIdx.x < size_a) {
        temp[threadIdx.x] = A[base + threadIdx.x];
    }
    if (threadIdx.x < 2 * RADIUS && base + THREADS_PER_BLOCK + threadIdx.x < size_a) {
        temp[threadIdx.x + THREADS_PER_BLOCK] = A[base + THREADS_PER_BLOCK + threadIdx.x];
    }
    __syncthreads();
    if (out_idx < size_b) {
        float sum = 0;
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
            sum += temp[threadIdx.x + dx + RADIUS];
        }
        B[out_idx] = sum;
    }
}

extern "C" {

void WindowSumSimple(float* in_array, float* out_array, int n) {
  int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  WindowSumSimpleKernel<<<num_blocks, THREADS_PER_BLOCK>>>(in_array, out_array, n);
}

void WindowSumShared(float* in_array, float* out_array, int size_a, int size_b) {
  int num_blocks = (size_b + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  WindowSumSharedKernel<<<num_blocks, THREADS_PER_BLOCK>>>(in_array, out_array, size_a, size_b);
  cudaDeviceSynchronize();
}
}