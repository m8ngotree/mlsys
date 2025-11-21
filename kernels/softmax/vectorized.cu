// https://maharshi.bearblog.dev/optimizing-softmax-cuda/

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_utils.cuh"

__global__ void softmax_kernel_4(float* __restrict__ xd, float* __restrict__ resd, int M, int N) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    float* input_row = xd + row * N;
    float* output_row = resd + row * N;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    int n_float4s = N / 4;
    int tail = N % 4;
    float4* input_row_vec = reinterpret_cast<float4*>(input_row);
    float4* output_row_vec = reinterpret_cast<float4*>(output_row);
    float maxval = -INFINITY;

    #pragma unroll
    for (int i = tid; i < n_float4s; i += blockDim.x) {
        float4 elem = input_row_vec[i];

        maxval = fmaxf(maxval, elem.x);
        maxval = fmaxf(maxval, elem.y);
        maxval = fmaxf(maxval, elem.z);
        maxval = fmaxf(maxval, elem.w);
        if (maxval > local_max) {
            local_norm *= __expf(local_max - maxval);
            local_max = maxval;
        }
        local_norm += __expf(elem.x - maxval);
        local_norm += __expf(elem.y - maxval);
        local_norm += __expf(elem.z - maxval);
        local_norm += __expf(elem.w - maxval);
    }

    if (tail && tid < tail) {
        float val = input_row[n_float4s * 4 + tid];
        if (val > local_max) {
            local_norm *= __expf(local_max - val);
            local_max = val;
        }
        local_norm += __expf(val - local_max);
    }
    __syncthreads();

    blockReduceMax<float>(local_max, smem, -INFINITY);
    __syncthreads();

    float row_max = smem[0];
    __syncthreads();

    float val = local_norm * expf(local_max - row_max);
    blockReduceSum<float>(val, smem, 0.0f);
    __syncthreads();

    float row_norm = smem[0];
    __syncthreads();

    #pragma unroll
    for (int i = tid; i < n_float4s; i += blockDim.x) {
        float4 elem = input_row_vec[i];
        elem.x = __expf(elem.x - row_max) / row_norm;
        elem.y = __expf(elem.y - row_max) / row_norm;
        elem.z = __expf(elem.z - row_max) / row_norm;
        elem.w = __expf(elem.w - row_max) / row_norm;

        output_row_vec[i] = elem;
    }
    if (tail && tid < tail)
    {
        float val = input_row[n_float4s * 4 + tid];
        output_row[n_float4s * 4 + tid] = __expf(val - row_max) / row_norm;
    }
}

float run_kernel_4(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    dim3 block_size(1024);
    dim3 grid_size(M);

    int warp_size = 32;
    size_t smem_size = CEIL_DIV(block_size.x, warp_size) * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    softmax_kernel_4<<<grid_size, block_size, smem_size>>>(matd, resd, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Kernel execution time: %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

