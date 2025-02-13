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
 * 
 * hint: define two matrices of size TILE_WIDTH x TILE_WIDTH in shared memory
 * sliding the tile along the matrix, compute partial sum of product.
 */
__global__ void MatMulTiledKernel(float* d_A, float* d_B, float* d_C, int N) {
    // define two matrices in share memory


    // define the row and column in the result matrix of current thread


    // iterate over tiles along row and column in d_A and d_B



    // store result

    
}