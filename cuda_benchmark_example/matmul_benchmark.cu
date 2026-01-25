

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

/*
For this demonstration, we are assuming that we are working with NxN matrices.
*/

using namespace std;

#define TILE_WIDTH 32


// Standard CPU Matmul
// O(N^3) TC
void Matmul(const float* A, const float *B, float* C, int N){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            float sum = 0.0f;
            for (int k = 0; k < N; k++){
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}



// Naive Matmul
// Follows similar pattern to CPU implementation...
__global__ void MM(const float *A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}



// Tiled Matmul
__global__ void TMM(const float* A, const float* B, float* C, int N){
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float CValue = 0.0f;


    for (int tile = 0; tile < (N + TILE_WIDTH - 1) / TILE_WIDTH; tile++ ){

        if (row < N && (tile * TILE_WIDTH + threadIdx.x) < N){
            As[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_WIDTH + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

         if (col < N && (tile * TILE_WIDTH + threadIdx.y) < N){
             Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_WIDTH + threadIdx.y) * N + col];
         } else {
             Bs[threadIdx.y][threadIdx.x] = 0.0f;
         }


         __syncthreads();
         for (int k = 0; k < TILE_WIDTH; ++k) {
             CValue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
         }
         __syncthreads();
    }

    if (row < N && col < N){
        C[row * N + col] = CValue;
    }

}


void run_benchmark(int N){
    size_t size = N * N * sizeof(float);
    vector<float> HA(N * N , 1.5f);
    vector<float> HB(N * N, 2.0f);
    vector<float> HC(N * N, 0.0f);

    float *DA, *DB, *DC;
    cudaMalloc(&DA, size);
    cudaMalloc(&DB, size);
    cudaMalloc(&DC, size);

    cudaMemcpy(DA, HA.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(DB, HB.data(), size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // GPU warmup
    MM<<<dimGrid, dimBlock>>>(DA, DB, DC, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    MM<<<dimGrid, dimBlock>>>(DA, DB, DC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naive_ms = 0.0f;
    cudaEventElapsedTime(&naive_ms, start, stop);


    cudaEventRecord(start);
    TMM<<<dimGrid, dimBlock>>>(DA, DB, DC, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float tile_ms = 0.0f;
    cudaEventElapsedTime(&tile_ms, start, stop);

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
    vector<int> sizes{512, 1024, 2048, 4096, 8192, 16384};
    for (int size : sizes){
        run_benchmark(size);
    }
    return 0;
}
