#ifndef COALESCED_LAYERNORM_CUH
#define COALESCED_LAYERNORM_CUH

#include <cuda_runtime.h>

__global__ void coalesced_layernorm_kernel(float *X, float *P, int m, int n);

float run_coalesced_ln(float *D_in, float *D_out, int m, int n);

#endif

