https://aryagxr.com/blogs/cuda-optimizing-layernorm

#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include "../include/layernorm_common.cuh"
#include "../include/naive_0.cuh"

__global__ void naive_layernorm_kernel(float *X, float *P, int m, int n) {
    int row = threadIdx.x + (blockDim.x * blockIdx.x);

    if (row < m) {
        float mean = 0.0f;
        float var = 0.0f;

        for (int col = 0; col < n; col++) {
            int idx = row * n + col;
            mean += X[idx];
        }
        mean /= n;

        for (int col = 0; col < n; col++) {
            int idx = row * n + col;
            var += (X[idx] - mean) * (X[idx] - mean);
        }
        var /= n;

        float stddev = sqrtf(var + EPSILON);
        for (int col = 0; col < n; col++) {
            int idx = row * n + col;
            P[idx] = (X[idx] - mean) / stddev;
        }
    }
}

float run_naive_ln(float *D_in, float *D_out, int m, int n) {
    dim3 threadsPerBlock(1024);
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    
    cudaEventRecord(start);
    naive_layernorm_kernel<<<blocksPerGrid, threadsPerBlock>>>(D_in, D_out, m, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Naive layernorm kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

