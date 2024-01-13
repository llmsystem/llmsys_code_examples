#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <fstream>


__global__ void MatmulKernel(const float* a, const float* b, float* out, 
                             int M, int N, int P) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= M * P) return;
  int row = idx / P;
  int col = idx % P;
  if (row < M && col < P) {
    float sum = 0.0;
    for (int i = 0; i < N; i++) {
      sum += a[row * N + i] * b[i * P + col];
    }
    out[row * P + col] = sum;
  }
}

extern "C" {

void Matmul(const float* a, const float* b, float* c, int M, int N, int P) {
    int n = M * P;
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    MatmulKernel<<<num_blocks, threads_per_block>>>(a, b, c, M, N, P);
}

}