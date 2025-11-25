https://aryagxr.com/blogs/cuda-optimizing-layernorm

#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include "../include/layernorm_common.cuh"
#include "../include/coalesced_1.cuh"

__global__ void coalesced_layernorm_kernel(float *X, float *P, int m, int n) {
    int col = threadIdx.x + (blockDim.x * blockIdx.x);

    if (col >= n) return;

    float mean = 0.0f;
    float var = 0.0f;

    for (int row = 0; row < m; row++) {
        int idx = col * m + row;
        mean += X[idx];
    }
    mean /= m;

    for (int row = 0; row < m; row++) {
        int idx = col * m + row;
        var += (X[idx] - mean) * (X[idx] - mean);
    }
    var /= m;

    float stddev = sqrtf(var + EPSILON);
    for (int row = 0; row < m; row++) {
        int idx = col * m + row;
        P[idx] = (X[idx] - mean) / stddev;
    }
}

float run_coalesced_ln(float *D_in, float *D_out, int m, int n) {
    dim3 threadsPerBlock(1024);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    
    cudaEventRecord(start);
    coalesced_layernorm_kernel<<<blocksPerGrid, threadsPerBlock>>>(D_in, D_out, m, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Coalesced layernorm kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

