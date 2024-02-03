#include <iostream>

using namespace std;
#define TILE_WIDTH 16


void naive_conv(int N, int H, int W, int K, int C_IN, int C_OUT, float *input, float *output, float *kernel) {
    int h_out = H - K + 1;
    int w_out = W - K + 1;  
    // kernel: C_OUT * C_IN * K * K
    // input: N * C_IN * H * W
    // output: N * C_OUT * h_out * w_out
    for(int n = 0; n < N; n++) {  // for each image in the mini-batch
        for(int c_in = 0; c_in < C_IN; c_in++) {  // for each output feature maps
            for(int c_out = 0; c_out < C_OUT; c_out++) {
                for(int h = 0; h < h_out; h++) {
                    for(int w = 0; w < w_out; w++) {
                        for(int i = 0; k < K; i++) {
                            for(int j = 0; j < K; j++) {
                                output[n, c_out, h, w] += input[n, c_in, h + i, w + j] * kernel[c_out, c_in, i, j];
                            }
                        }
                    }
                }
            }
        }
    }
}

void unroll_conv(int N, int H, int W, int K, int C_IN, int C_OUT, float *input, float *output, float *kernel) {
    int h_out = H - K + 1;
    int w_out = W - K + 1;  
    int W_unroll = C_IN * K * K;
    int H_unroll = h_out * w_out;
    float* input_unroll = new float[W_unroll * H_unroll];
    for(int i = 0; i < N; i++) {
        unroll(input[i], input_unroll, C_IN, H, W, K, h_out, w_out);
        gemm(input_unroll, kernel, output[i], W_unroll, C_OUT, H_unroll);
    }
}

void im2col(float* input, int C_IN, int H, int W, int K, float* output) {
    int h_out = H - K + 1;
    int w_out = W - K + 1;
    int h_unroll = C_IN * K * K;
    int w_unroll = h_out * w_out;

    for (int c = 0; c < C_IN; ++c) {
        for(int h = 0; h < h_out; h++) {
            for(int w = 0; w < w_out; w++) {
                for(int i = 0; i < K; i++) {
                    for(int j = 0; j < K; j++) {
                        output[c * K * K + (h * w_out + w)][i * K + j] = input[c * H * W + (h + i) * W + w + j];
                        output[c * K * K * h_out * w_out + i * K * w_out + j * w_out + h * w_out + w] = input[c, h + i, w + j];
                    }
                }
            }
        }
    }
}

void unroll(float *input, float *input_unroll, int C_IN, int H, int W, int K, int h_out, int w_out) {
    for(int c_in = 0; c_in < C_IN; c_in++) {
        int w_base = c_in * K * K;
        for(int h = 0; h < h_out; h++) {
            for(int w = 0; w < w_out; w++) {
                for(int i = 0; i < K; i++) {
                    for(int j = 0; j < K; j++) {
                        input_unroll[w_base * h_out * w_out + h * w_out + w] = input[c_in * H * W + (h + i) * W + w + j];
                    }
                }
            }
        }
    }
}