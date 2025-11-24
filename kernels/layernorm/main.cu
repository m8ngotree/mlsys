#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <stdlib.h>

#include "include/naive_0.cuh"
#include "include/coalesced_1.cuh"
#include "include/smem_2.cuh"
#include "include/shfl_3.cuh"
#include "include/vectorized_4.cuh"

#include "kernels/naive_0.cu"
#include "kernels/coalesced_1.cu"
#include "kernels/smem_2.cu"
#include "kernels/shfl_3.cu"
#include "kernels/vectorized_4.cu"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main(int argc, char *argv[]) {
    int kernel_choice = 4;
    if (argc > 1) {
        kernel_choice = atoi(argv[1]);
    }

    int M = 1024;
    int N = 1024;

    if (argc > 2) {
        M = atoi(argv[2]);
    }
    if (argc > 3) {
        N = atoi(argv[3]);
    }

    size_t matrix_size = M * N * sizeof(float);
    float *X_input, *P_output;
    float *D_input, *D_output;

    X_input = (float *)malloc(matrix_size);
    P_output = (float *)malloc(matrix_size);

    for (int i = 0; i < M * N; i++) {
        X_input[i] = i + 1;
    }

    CUDA_CHECK(cudaMalloc((void **)&D_input, matrix_size));
    CUDA_CHECK(cudaMalloc((void **)&D_output, matrix_size));

    CUDA_CHECK(cudaMemcpy(D_input, X_input, matrix_size, cudaMemcpyHostToDevice));

    printf("Running layernorm with M=%d, N=%d\n", M, N);
    printf("Kernel choice: %d (0=naive, 1=coalesced, 2=smem, 3=shfl, 4=vectorized)\n", kernel_choice);

    float ms = 0.0f;
    switch (kernel_choice) {
        case 0:
            ms = run_naive_ln(D_input, D_output, M, N);
            break;
        case 1:
            printf("Warning: Coalesced kernel expects column-major layout!\n");
            ms = run_coalesced_ln(D_input, D_output, M, N);
            break;
        case 2:
            ms = run_smem_ln(D_input, D_output, M, N);
            break;
        case 3:
            ms = run_shfl_ln(D_input, D_output, M, N);
            break;
        case 4:
        default:
            ms = run_vect_ln(D_input, D_output, M, N);
            break;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(P_output, D_output, matrix_size, cudaMemcpyDeviceToHost));

    printf("\nPer-row verification (first 5 rows):\n");
    for (int row = 0; row < 5 && row < M; row++) {
        float row_sum = 0.0f;
        float row_sum_sq = 0.0f;
        
        for (int col = 0; col < N; col++) {
            float val = P_output[row * N + col];
            row_sum += val;
            row_sum_sq += val * val;
        }
        
        float row_mean = row_sum / N;
        float row_var = (row_sum_sq / N) - (row_mean * row_mean);
        float row_std = sqrtf(row_var);
        
        printf("Row %d - Mean: %.6f, Std: %.6f\n", row, row_mean, row_std);
    }

    free(P_output);
    free(X_input);
    cudaFree(D_input);
    cudaFree(D_output);

    return 0;
}