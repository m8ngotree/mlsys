#ifndef NAIVE_LAYERNORM_CUH
#define NAIVE_LAYERNORM_CUH

#include <cuda_runtime.h>

__global__ void naive_layernorm_kernel(float *X, float *P, int m, int n);

float run_naive_ln(float *D_in, float *D_out, int m, int n);

#endif

