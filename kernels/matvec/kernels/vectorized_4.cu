https://maharshi.bearblog.dev/optimizing-sgemv-cuda/

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "utils.cuh"

__global__ void vectorized_sgemv_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    extern __shared__ float smem[];

    int bid = blockIdx.x;
    if (bid >= M) return;

    int tid = threadIdx.x;
    int n_float4s = N / 4;

    float4* mat_row = reinterpret_cast<float4*>(matd + bid * N);
    float4* vec = reinterpret_cast<float4*>(vecd);

    float partial_sum = 0.f;

#pragma unroll 4
    for (int col = tid; col < n_float4s; col += blockDim.x) {
        float4 matval = mat_row[col];
        float4 vecval = vec[col];

        partial_sum += (matval.x * vecval.x +
                        matval.y * vecval.y +
                        matval.z * vecval.z +
                        matval.w * vecval.w);
    }

    blockReduceSum(partial_sum, smem, tid, blockDim.x);
    if (tid == 0) {
        float sum = smem[0];
        resd[bid] = sum;
    }
}

float run_kernel_vectorized_sgmev(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N, float THEORETICAL_MAX_GFLOPS, float THEORETICAL_MAX_MEMORY_BANDWIDTH) {
    int NUM_THREADS = 64;
    int warp_size = 32;

    dim3 block_size(NUM_THREADS);
    dim3 grid_size(M);
    size_t shared_mem_size = CEIL_DIV(block_size.x, warp_size) * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    vectorized_sgemv_kernel<<<grid_size, block_size, shared_mem_size>>>(matd, vecd, resd, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("------- Vectorized sgmev kernel ---------\n");
    print_kernel_essentials(M, N, ms, THEORETICAL_MAX_GFLOPS, THEORETICAL_MAX_MEMORY_BANDWIDTH);
    printf("---------------------------\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

