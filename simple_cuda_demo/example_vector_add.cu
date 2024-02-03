#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <fstream>

__global__ void VecAddKernel(int* A, int* B, int* C, int n) {
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