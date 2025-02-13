#include <cuda_runtime.h>
#include <iostream>

using namespace std;
#define TILE_WIDTH 16

void matrix_multiply(float **a, float **b, float **c, float N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            c[i][j] = 0;
            for (int k = 0; k < N; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

/**
 * @brief compute C=A*B using tile size TILE_WIDTH
 * 
 * @param d_A matrix A
 * @param d_B matrix B
 * @param d_C result matrix C
 * @param N size of matrix (number of rows and columns)
 */
__global__ void MatMulTiledKernel(float* d_A, float* d_B, float* d_C, int N) {
	__shared__ float As[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

	// Determine the row and col of the P element to be calculated for the thread
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float Cvalue = 0;
	for(int ph = 0; ph < N/TILE_WIDTH; ++ph) {
		As[threadIdx.y][threadIdx.x] = d_A[row * N + ph * TILE_WIDTH + threadIdx.x];
		Bs[threadIdx.y][threadIdx.x] = d_B[(ph * TILE_WIDTH + threadIdx.y) * N + col];
		__syncthreads();
		for(int k = 0; k < TILE_WIDTH; ++k) {
			Cvalue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
		}
		__syncthreads();
	}
	d_C[row * N + col] = Cvalue;
}