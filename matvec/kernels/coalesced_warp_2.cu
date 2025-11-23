#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "utils.cuh"

__global__ void coalesced_warp_sgmev_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    assert(blockDim.x == warpSize);

    int bid = blockIdx.x;
    if (bid >= M) return;

    int tid = threadIdx.x;
    float partial_sum = 0.f;
    for (int col = tid; col < N; col += blockDim.x) {
        partial_sum += matd[bid * N + col] * vecd[col];
    }

    float sum = warpReduceSum(partial_sum);
    if (tid == 0) {
        resd[bid] = sum;
    }
}

float run_kernel_coalesced_warp_sgmev(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N, float THEORETICAL_MAX_GFLOPS, float THEORETICAL_MAX_MEMORY_BANDWIDTH) {
    int NUM_THREADS = 32;

    dim3 block_size(NUM_THREADS);
    dim3 grid_size(M);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    coalesced_warp_sgmev_kernel<<<grid_size, block_size>>>(matd, vecd, resd, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("------- Coalesced warp sgmev kernel ---------\n");
    print_kernel_essentials(M, N, ms, THEORETICAL_MAX_GFLOPS, THEORETICAL_MAX_MEMORY_BANDWIDTH);
    printf("---------------------------\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

