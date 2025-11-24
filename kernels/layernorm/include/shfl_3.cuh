#ifndef SHFL_LAYERNORM_CUH
#define SHFL_LAYERNORM_CUH

#include <cuda_runtime.h>

__global__ void shfl_layernorm_kernel(float *X, float *P, int m, int n);

float run_shfl_ln(float *D_in, float *D_out, int m, int n);

#endif

