#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <vector>

using namespace std;
using std::generate;
using std::vector;

__global__ void VecAddKernel(int* A, int* B, int* C, int n) {
  // blockDim is size of block along x-axis
  // blockIdx is the index of the current thread's block
  // threadIdx is the index of the current thread within the block
  // Compute the global thread ID
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    // Calculate the addition of the ith element of A and B
    C[i] = A[i] + B[i];
  }
}


extern "C" {

void VecAddCPU(int* A, int* B, int* C, int n) {
  for(int i = 0; i < n; ++i) {
    C[i] = A[i] + B[i];
  }
}


void VecAddCUDA(int* Agpu, int* Bgpu, int* Cgpu, int n) {
  // In this example, we load the data into the GPU by Python codes.
  int threads_per_block = 256;
  int num_blocks = (n + threads_per_block - 1) / threads_per_block;
  VecAddKernel<<<num_blocks, threads_per_block>>>(Agpu, Bgpu, Cgpu, n);
}


void VecAddCUDA2(int* Acpu, int* Bcpu, int* Ccpu, int n) {
  // In this example, we load the data into the GPU by C++ codes.
  int *dA, *dB, *dC;
  // Allocate device memory
  cudaMalloc(&dA, n * sizeof(int));
  cudaMalloc(&dB, n * sizeof(int));
  cudaMalloc(&dC, n * sizeof(int));
  // Copy data from host memory to device memory
  cudaMemcpy(dA, Acpu, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, Bcpu, n * sizeof(int), cudaMemcpyHostToDevice);
  // Launch the CUDA kernel
  int threads_per_block = 256;
  int num_blocks = (n + threads_per_block - 1) / threads_per_block;
  VecAddKernel<<<num_blocks, threads_per_block>>>(dA, dB, dC, n);
  // Copy the result from device memory to host memory
  cudaMemcpy(Ccpu, dC, n * sizeof(int), cudaMemcpyDeviceToHost);
  // Free device memory
  cudaFree(dA); 
  cudaFree(dB); 
  cudaFree(dC);
}

}

// Check result on the CPU
void verify_result_add(vector<int> &a, vector<int> &b, vector<int> &c, int N) {
  // For every row...
  for (int i = 0; i < N; i++) {
    // For every column...
    for (int j = 0; j < N; j++) {
      // For every element in the row-column pair
      // Check against the CPU result
      if (a[i * N + j] + b[i * N + j] != c[i * N + j]) {
        printf("Error in (%d, %d): %d + %d != %d\n", i, j, a[i * N + j], b[i * N + j], c[i * N + j]);
      }
      assert(a[i * N + j] + b[i * N + j] == c[i * N + j]);
    }
  }
}

int main() {
  // length of the vector
  int n = 1024;

  // Host vectors
  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);

  // Initialize matrices
  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  VecAddCUDA2(h_a.data(), h_b.data(), h_c.data(), n);

  cudaDeviceSynchronize();

  // Check result
  verify_result_add(h_a, h_b, h_c, N);

  cout << "COMPLETED SUCCESSFULLY\n";
}