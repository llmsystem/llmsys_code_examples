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


__global__ void MatMulTiledKernel(float* d_M, float* d_N, float* d_P, int N) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;
    for(int ph = 0; ph < N/TILE_WIDTH; ++ph) {
        Mds[ty][tx] = d_M[row * N + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = d_N[(ph * TILE_WIDTH + ty) * N + col];
        __syncthreads();
        for(int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    d_P[row * N + col] = Pvalue;
}