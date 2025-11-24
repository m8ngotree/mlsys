#ifndef SMEM_LAYERNORM_CUH
#define SMEM_LAYERNORM_CUH

#include <cuda_runtime.h>

__global__ void smem_layernorm_kernel(float *X, float *P, int m, int n);

float run_smem_ln(float *D_in, float *D_out, int m, int n);

#endif

