// https://aryagxr.com/blogs/cuda-optimizing-layernorm

#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include "../include/layernorm_common.cuh"
#include "../include/vectorized_4.cuh"

__global__ void vectorized_layernorm_kernel(float *X, float *P, int m, int n) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tidx = threadIdx.x;

    if (row >= m) return;

    float *row_in = X + row * n;
    float *row_out = P + row * n;
    float lmean = 0.0f;
    float lvar = 0.0f;

    int vec_iters = n / 4;

    for (int i = tidx; i < vec_iters; i += blockDim.x) {
        float4 v = reinterpret_cast<float4 *>(row_in)[i];
        lmean += v.x + v.y + v.z + v.w;
        lvar += (v.x * v.x) + (v.y * v.y) + (v.z * v.z) + (v.w * v.w);
    }

    for (int i = vec_iters * 4 + tidx; i < n; i += blockDim.x) {
        float a = row_in[i];
        lmean += a;
        lvar += (a * a);
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        lmean += __shfl_down_sync(0xffffffff, lmean, offset);
        lvar += __shfl_down_sync(0xffffffff, lvar, offset);
    }

    if (blockDim.x > WARP_SIZE) {
        if (tidx % WARP_SIZE == 0) {
            int warp_id = tidx / WARP_SIZE;
            smem[warp_id] = lmean;
            smem[warp_id + blockDim.x / WARP_SIZE] = lvar;
        }
        __syncthreads();

        if (tidx < WARP_SIZE) {
            int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
            lmean = (tidx < num_warps) ? smem[tidx] : 0.0f;
            lvar = (tidx < num_warps) ? smem[tidx + num_warps] : 0.0f;

            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                lmean += __shfl_down_sync(0xffffffff, lmean, offset);
                lvar += __shfl_down_sync(0xffffffff, lvar, offset);
            }

            if (tidx == 0) {
                smem[0] = lmean;
                smem[1] = lvar;
            }
        }
    } else {
        if (tidx == 0) {
            smem[0] = lmean;
            smem[1] = lvar;
        }
    }
    __syncthreads();

    float gmean = smem[0] / n;
    float gvar = (smem[1] / n) - (gmean * gmean);
    gvar = fmaxf(gvar, 0.0f);
    float std_inv = rsqrtf(gvar + EPSILON);

    for (int i = tidx; i < vec_iters; i += blockDim.x) {
        float4 v = reinterpret_cast<float4 *>(row_in)[i];
        v.x = (v.x - gmean) * std_inv;
        v.y = (v.y - gmean) * std_inv;
        v.z = (v.z - gmean) * std_inv;
        v.w = (v.w - gmean) * std_inv;
        reinterpret_cast<float4 *>(row_out)[i] = v;
    }

    for (int i = vec_iters * 4 + tidx; i < n; i += blockDim.x) {
        row_out[i] = (row_in[i] - gmean) * std_inv;
    }
}

float run_vect_ln(float *D_in, float *D_out, int m, int n) {
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(m);

    int num_warps = (threadsPerBlock.x + WARP_SIZE - 1) / WARP_SIZE;
    int smem_size = (num_warps * 2) * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    
    cudaEventRecord(start);
    vectorized_layernorm_kernel<<<blocksPerGrid, threadsPerBlock, smem_size>>>(D_in, D_out, m, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Vectorized layernorm kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}