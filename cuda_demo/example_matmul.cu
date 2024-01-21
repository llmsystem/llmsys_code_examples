#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <fstream>


__global__ void MatmulKernel(const float* a, const float* b, float* out, 
                             int M, int N, int P) {
  // Calculate the global thread index and the row and column it corresponds to
  // Every thread will compute one element of the output matrix
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int row = idx / P;
  int col = idx % P;
  // Compute the summation of the dot product of the row of a and the column of b
  if (row < M && col < P) {
    float sum = 0.0;
    for (int i = 0; i < N; i++) {
      sum += a[row * N + i] * b[i * P + col];
    }
    out[row * P + col] = sum;
  }
}

extern "C" {

// This functions takes in arrays which are already on the GPU
// and will return arrays which are also on the GPU
// Copying values between the device memory and host memory is done in the python codes

void Matmul(const float* a, const float* b, float* c, int M, int N, int P) {
    int n = M * P;
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    MatmulKernel<<<num_blocks, threads_per_block>>>(a, b, c, M, N, P);
}

}