#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>

using namespace std;
#define TILE_WIDTH 32

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
__global__ void matMulTiled(const float* d_A, const float* d_B, float* d_C, int N) {
    // define two matrices in share memory


    // define the row and column in the result matrix of current thread


    // iterate over tiles along row and column in d_A and d_B



    // store result

    
}

/**
 * simple matrix multiplication kernel (no tiling)
 * 
 * @param a 
 * @param b 
 * @param c 
 * @param N 
 */
__global__ void matMul(const float *a, const float *b, float *c, int N) {
  // Compute each thread's global row and column index
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= N || col >= N) return;
  // Iterate over row, and down column
  c[row * N + col] = 0;
  for (int k = 0; k < N; k++) {
    // Accumulate results for a single element
    c[row * N + col] += a[row * N + k] * b[k * N + col];
  }
}

void run_benchmark(int N){
    size_t size = N * N * sizeof(float);
    vector<float> HA(N * N , 0.0f);
    vector<float> HB(N * N, 0.0f);
    vector<float> HC(N * N, 0.0f);

    static mt19937 gen{random_device{}()}; 
    uniform_real_distribution<float> dis(-1, 1);
    generate(HA.begin(), HA.end(), [&dis]() { return dis(gen); });
    generate(HB.begin(), HB.end(), [&dis]() { return dis(gen); });

    float *DA, *DB, *DC;
    cudaMalloc(&DA, size);
    cudaMalloc(&DB, size);
    cudaMalloc(&DC, size);

    cudaMemcpy(DA, HA.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(DB, HB.data(), size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // GPU warmup
    matMul<<<dimGrid, dimBlock>>>(DA, DB, DC, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMul<<<dimGrid, dimBlock>>>(DA, DB, DC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naive_ms = 0.0f;
    cudaEventElapsedTime(&naive_ms, start, stop);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
    }

    cudaEventRecord(start);
    matMulTiled<<<dimGrid, dimBlock>>>(DA, DB, DC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float tile_ms = 0.0f;
    cudaEventElapsedTime(&tile_ms, start, stop);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
    }

    // output to input into python later
    cout << N << "," << naive_ms << "," << tile_ms << endl;

    cudaFree(DA);
    cudaFree(DB);
    cudaFree(DC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(){
    cout<<"Naive,Tiled"<<endl;
    vector<int> sizes{512, 1024, 2048, 4096, 8192};
    for (int size : sizes){
        run_benchmark(size);
    }
    return 0;
}