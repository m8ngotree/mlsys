#ifndef VECTORIZED_LAYERNORM_CUH
#define VECTORIZED_LAYERNORM_CUH

#include <cuda_runtime.h>

__global__ void vectorized_layernorm_kernel(float *X, float *P, int m, int n);

float run_vect_ln(float *D_in, float *D_out, int m, int n);

#endif

