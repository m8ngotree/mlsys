#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include "../include/layernorm_common.cuh"
#include "../include/shfl_3.cuh"

__global__ void shfl_layernorm_kernel(float *X, float *P, int m, int n) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tidx = threadIdx.x;

    if (row < m) {
        float *row_in = X + row * n;
        float *row_out = P + row * n;

        float lmean = 0.0f;
        float lvar = 0.0f;

        for (int i = tidx; i < n; i += blockDim.x) {
            float a = row_in[i];
            lmean += a;
            lvar += (a * a);
        }

        float lrmean = lmean;

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            lrmean += __shfl_down_sync(0xffffffff, lrmean, offset);
        }

        if (blockDim.x > WARP_SIZE) {
            if (tidx % WARP_SIZE == 0) {
                smem[tidx / WARP_SIZE] = lrmean;
            }
            __syncthreads();

            if (tidx < WARP_SIZE) {
                lrmean = (tidx < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? smem[tidx] : 0.0f;
                for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                    lrmean += __shfl_down_sync(0xffffffff, lrmean, offset);
                }

                if (tidx == 0) {
                    smem[0] = lrmean;
                }
            }
        } else {
            if (tidx == 0) {
                smem[0] = lrmean;
            }
        }
        __syncthreads();

        float gmean = smem[0] / n;

        float lrvar = lvar;

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            lrvar += __shfl_down_sync(0xffffffff, lrvar, offset);
        }

        if (blockDim.x > WARP_SIZE) {
            if (tidx % WARP_SIZE == 0) {
                smem[tidx / WARP_SIZE] = lrvar;
            }
            __syncthreads();

            if (tidx < WARP_SIZE) {
                lrvar = (tidx < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? smem[tidx] : 0.0f;
                for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                    lrvar += __shfl_down_sync(0xffffffff, lrvar, offset);
                }

                if (tidx == 0) {
                    smem[0] = lrvar;
                }
            }
        } else {
            if (tidx == 0) {
                smem[0] = lrvar;
            }
        }
        __syncthreads();

        float gvar = (smem[0] / n) - (gmean * gmean);

        gvar = fmaxf(gvar, 0.0f);
        float stddev = rsqrtf(gvar + EPSILON);

        for (int i = tidx; i < n; i += blockDim.x) {
            row_out[i] = (row_in[i] - gmean) * stddev;
        }
    }
}

float run_shfl_ln(float *D_in, float *D_out, int m, int n) {
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(m);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    
    cudaEventRecord(start);
    int smem_size = ((threadsPerBlock.x + WARP_SIZE - 1) / WARP_SIZE + 1) * sizeof(float);
    shfl_layernorm_kernel<<<blocksPerGrid, threadsPerBlock, smem_size>>>(D_in, D_out, m, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Warp shuffle layernorm kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}