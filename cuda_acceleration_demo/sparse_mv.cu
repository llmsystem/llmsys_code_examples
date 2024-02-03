#include <cuda_runtime.h>
#include <iostream>

using namespace std;
#define TILE_WIDTH 16

void spmv_csr(float *data, int *col_index, int *row_ptr, float *x, float *y, int n) {
    for(int row = 0; row < n; row++) {
        float dot = 0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for(int elem = row_start; elem < row_end; elem++) {
            dot += x[row] * data[col_index[elem]];
        }
        y[row] += dot;
    }
}

__global__ void SpMVCSRKernel(float *data, int *col_index, int *row_ptr, float *x, float *y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < num_rows) {
        float dot = 0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for(int elem = row_start; elem < row_end; elem++) {
            dot += x[row] * data[col_index[elem]];
        }
        y[row] += dot;
    }
}