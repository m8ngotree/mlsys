// https://maharshi.bearblog.dev/optimizing-softmax-cuda/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_utils.cuh"

void run_kernel_0(float* __restrict__ matd, float* __restrict__ resd, int M, int N);
void run_kernel_1(float* __restrict__ matd, float* __restrict__ resd, int M, int N);
void run_kernel_2(float* __restrict__ matd, float* __restrict__ resd, int M, int N);
void run_kernel_3(float* __restrict__ matd, float* __restrict__ resd, int M, int N);
float run_kernel_4(float* __restrict__ matd, float* __restrict__ resd, int M, int N);

float random_normal_clamped(float min, float max) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    if (num < min)
        return min;
    if (num > max)
        return max;
    return num;
}

int main() {
    int M = 1024;
    int N = 32768;
    int matsize = M * N;
    int totalsize = matsize * sizeof(float);

    float* mat = (float*)malloc(totalsize);
    float* res = (float*)malloc(totalsize);
    for (int i = 0; i < matsize; i++) {
        mat[i] = random_normal_clamped(-10, 10);
    }

    float *matd, *resd;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&matd, totalsize));
    CUDA_CHECK(cudaMalloc(&resd, totalsize));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> GPU allocation time: %f ms\n", ms);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(matd, mat, totalsize, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Host to device transfer time: %f ms\n", ms);

    printf("\n========== Testing Softmax Kernels (M=%d, N=%d) ==========\n", M, N);
    
    printf("\n--- Kernel 0: Naive (3 passes, one thread per row) ---\n");
    run_kernel_0(matd, resd, M, N);

    printf("\n--- Kernel 1: Online (2 passes, one thread per row) ---\n");
    run_kernel_1(matd, resd, M, N);

    printf("\n--- Kernel 2: Shared Memory (parallel reduction, one block per row) ---\n");
    run_kernel_2(matd, resd, M, N);

    printf("\n--- Kernel 3: Warp Shuffle (warp-level primitives for reduction) ---\n");
    run_kernel_3(matd, resd, M, N);

    printf("\n--- Kernel 4: Vectorized (float4 loads + warp shuffle primitives) ---\n");
    run_kernel_4(matd, resd, M, N);

    printf("\n========== All kernels completed ==========\n\n");

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(res, resd, totalsize, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Device to host transfer time: %f ms\n", ms);

    free(mat);
    free(res);
    CUDA_CHECK(cudaFree(matd));
    CUDA_CHECK(cudaFree(resd));

    return 0;
}

