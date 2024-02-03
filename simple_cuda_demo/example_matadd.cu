// This program computes a simple version of matrix multiplication
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;
using std::generate;
using std::vector;


__global__ void matrixAdd(const int * a, const int * b,
                          int * c, int N) {
  // Compute each thread's global row and column index
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  
  // Iterate over row, and down column
  if (row < N && col < N) {
    c[row * N + col] = a[row * N + col] + b[row * N + col];
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
  // Matrix size of 1024 x 1024;
  int N = 1 << 10;

  // Size (in bytes) of matrix
  size_t bytes = N * N * sizeof(int);

  // Host vectors
  vector<int> h_a(N * N);
  vector<int> h_b(N * N);
  vector<int> h_c(N * N);

  // Initialize matrices
  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  
  // Allocate device memory
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);


  // Copy data to the device
  cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);


  // Threads per CTA dimension
  int THREADS = 32;

  // Blocks per grid dimension (assumes THREADS divides N evenly)
  int BLOCKS = N / THREADS;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);  // should be <= 1024
  dim3 blocks(BLOCKS, BLOCKS);

  // Launch kernel
  matrixAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);

  // Copy back to the host
  cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
  }

  // Check result
  verify_result_add(h_a, h_b, h_c, N);

  cout << "COMPLETED SUCCESSFULLY\n";

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
