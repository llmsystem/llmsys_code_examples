// This program computes a simple version of matrix multiplication

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

using std::cout;
using std::generate;
using std::vector;

__global__ void matrixMul(const int *a, const int *b, int *c, int M, int N, int P) {
  // Compute each thread's global row and column index
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= P) return;
  // Iterate over row, and down column
  c[row * P + col] = 0;
  for (int k = 0; k < N; k++) {
    // Accumulate results for a single element
    c[row * P + col] += a[row * N + k] * b[k * P + col];
  }
}

// Check result on the CPU
void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, int M, int N, int P) {
  // For every row...
  for (int i = 0; i < M; i++) {
    // For every column...
    for (int j = 0; j < P; j++) {
      // For every element in the row-column pair
      int tmp = 0;
      for (int k = 0; k < N; k++) {
        // Accumulate the partial results
        tmp += a[i * N + k] * b[k * P + j];
      }

      // Check against the CPU result
      if (tmp != c[i * P + j]) {
        printf("Error in (%d, %d): %d != %d\n", i, j, tmp, c[i * P + j]);
      }
      assert(tmp == c[i * P + j]);
    }
  }
}

int main() {
  // Matrix size of 256 x 1024, 1024 x 512;
  int M = 1 << 8;
  int N = 1 << 10;
  int P = 1 << 9;


  // Host vectors
  vector<int> h_a(M * N);
  vector<int> h_b(N * P);
  vector<int> h_c(M * P);

  // Initialize matrices
  generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
  generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

  // Allocate device memory
  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, M * N * sizeof(int));
  cudaMalloc(&d_b, N * P * sizeof(int));
  cudaMalloc(&d_c, M * P * sizeof(int));

  // Copy data to the device
  cudaMemcpy(d_a, h_a.data(), M * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), N * P * sizeof(int), cudaMemcpyHostToDevice);

  // Threads per CTA dimension
  int THREADS = 32;

  // Blocks per grid dimension (assumes THREADS divides N evenly)
  int BLOCKS_X = M / THREADS, BLOCKS_Y = P / THREADS;

  // Use dim3 structs for block  and grid dimensions
  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS_X, BLOCKS_Y);

  // Launch kernel
  matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, M, N, P);

  // Copy back to the host
  cudaMemcpy(h_c.data(), d_c, M * P * sizeof(int), cudaMemcpyDeviceToHost);

  // Check result
  verify_result(h_a, h_b, h_c, M, N, P);

  cout << "COMPLETED SUCCESSFULLY\n";

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
