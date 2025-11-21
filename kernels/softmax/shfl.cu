// https://maharshi.bearblog.dev/optimizing-softmax-cuda/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_utils.cuh"

__global__ void softmax_kernel_3(float* xd, float* resd, int M, int N) {
    __shared__ float smem[1024];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    unsigned int warp_size = 32;
    if (row >= M) return;

    float* input_row = xd + row * N;
    float* output_row = resd + row * N;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        float x = input_row[i];
        if (x > local_max) {
            local_norm *= expf(local_max - x);
            local_max = x;
        }
        local_norm += expf(x - local_max);
    }
    __syncthreads();

    float val = local_max;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    if (blockDim.x > warp_size) {
        if (tid % warp_size == 0) {
            smem[tid / warp_size] = val;
        }
        __syncthreads();

        if (tid < warp_size) {
            val = (tid < CEIL_DIV(blockDim.x, warp_size)) ? smem[tid] : -INFINITY;
            for (int offset = warp_size / 2; offset > 0; offset /= 2) {
                val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
            }
            if (tid == 0) smem[0] = val;
        }
    } else {
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();

    float row_max = smem[0];
    __syncthreads();

    val = local_norm * expf(local_max - row_max);
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (blockDim.x > warp_size) {
        if (tid % warp_size == 0) {
            smem[tid / warp_size] = val;
        }
        __syncthreads();

        if (tid < warp_size) {
            val = (tid < CEIL_DIV(blockDim.x, warp_size)) ? smem[tid] : 0.0f;
            for (int offset = warp_size / 2; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (tid == 0) smem[0] = val;
        }
    } else {
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();

    float row_norm = smem[0];
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        output_row[i] = expf(input_row[i] - row_max) / row_norm;
    }
}

void run_kernel_3(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    dim3 block_size(1024);
    dim3 grid_size(M);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    softmax_kernel_3<<<grid_size, block_size>>>(matd, resd, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Kernel execution time: %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

