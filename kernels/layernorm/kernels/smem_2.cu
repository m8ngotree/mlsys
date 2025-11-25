https://aryagxr.com/blogs/cuda-optimizing-layernorm

#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include "../include/layernorm_common.cuh"
#include "../include/smem_2.cuh"

__global__ void smem_layernorm_kernel(float *X, float *P, int m, int n) {
    __shared__ float smem[1024];

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

        smem[tidx] = lmean;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tidx < stride) {
                smem[tidx] += smem[tidx + stride];
            }
            __syncthreads();
        }

        float gmean = smem[0] / n;

        smem[tidx] = lvar;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (tidx < stride) {
                smem[tidx] += smem[tidx + stride];
            }
            __syncthreads();
        }
        float gvar = (smem[0] / n) - (gmean * gmean);
        float stddev = rsqrtf(gvar + EPSILON);

        for (int i = tidx; i < n; i += blockDim.x) {
            row_out[i] = (row_in[i] - gmean) * stddev;
        }
    }
}

float run_smem_ln(float *D_in, float *D_out, int m, int n) {
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(m);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.f;
    
    cudaEventRecord(start);
    smem_layernorm_kernel<<<blocksPerGrid, threadsPerBlock>>>(D_in, D_out, m, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Shared memory layernorm kernel execution time: " << ms << " ms\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

