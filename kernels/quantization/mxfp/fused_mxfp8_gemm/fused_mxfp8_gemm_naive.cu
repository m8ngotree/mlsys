#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16
#define THREAD_TILE_M 4
#define THREAD_TILE_N 4

__device__ __forceinline__ float dequantize_mxfp8(uint8_t mxfp8_val, uint8_t scale) {
    int mantissa = mxfp8_val & 0x7F;
    int sign = (mxfp8_val >> 7) ? -1 : 1;
    
    int scale_exp = scale - 127;
    float scale_val = powf(2.0f, (float)scale_exp);
    
    float normalized = 1.0f + (float)mantissa / 128.0f;
    
    return sign * normalized * scale_val;
}

__global__ void fused_mxfp8_gemm_naive(
    const uint8_t* A_mxfp8,
    const uint8_t* A_scales,
    const uint8_t* B_mxfp8,
    const uint8_t* B_scales,
    float* C,
    int M, int N, int K
) {
    __shared__ uint8_t A_shared[TILE_M][TILE_K];
    __shared__ uint8_t B_shared[TILE_K][TILE_N];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * 16 + tx;
    
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;
    
    int C_row_start = block_row * TILE_M + tx * THREAD_TILE_M;
    int C_col_start = block_col * TILE_N + ty * THREAD_TILE_N;
    
    float C_reg[THREAD_TILE_M][THREAD_TILE_N] = {0};
    
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int A_row_start = block_row * TILE_M;
        int A_col_start = k_tile * TILE_K;
        int B_row_start = k_tile * TILE_K;
        int B_col_start = block_col * TILE_N;
        
        // Load A_tile
        for (int i = 0; i < 4; i++) {
            int elem_id = tid * 4 + i;
            int local_row = elem_id / TILE_K;
            int local_col = elem_id % TILE_K;
            
            int global_row = A_row_start + local_row;
            int global_col = A_col_start + local_col;
            
            if (global_row < M && global_col < K) {
                A_shared[local_row][local_col] = A_mxfp8[global_row * K + global_col];
            } else {
                A_shared[local_row][local_col] = 0;
            }
        }
        
        // Load B_tile
        for (int i = 0; i < 4; i++) {
            int elem_id = tid * 4 + i;
            int local_row = elem_id / TILE_N;
            int local_col = elem_id % TILE_N;
            
            int global_row = B_row_start + local_row;
            int global_col = B_col_start + local_col;
            
            if (global_row < K && global_col < N) {
                B_shared[local_row][local_col] = B_mxfp8[global_row * N + global_col];
            } else {
                B_shared[local_row][local_col] = 0;
            }
        }
        
        __syncthreads();
        
        // Compute
        for (int k = 0; k < TILE_K; k++) {
            int global_k = k_tile * TILE_K + k;
            if (global_k >= K) break;
            
            int scale_k_idx = global_k / 32;
            
            float a_vals[THREAD_TILE_M];
            for (int i = 0; i < THREAD_TILE_M; i++) {
                int local_row = tx * THREAD_TILE_M + i;
                int global_row = C_row_start + i;
                
                if (global_row < M) {
                    uint8_t a_mxfp8 = A_shared[local_row][k];
                    uint8_t a_scale = A_scales[global_row * (K/32) + scale_k_idx];
                    a_vals[i] = dequantize_mxfp8(a_mxfp8, a_scale);
                } else {
                    a_vals[i] = 0.0f;
                }
            }
            
            float b_vals[THREAD_TILE_N];
            for (int j = 0; j < THREAD_TILE_N; j++) {
                int local_col = ty * THREAD_TILE_N + j;
                int global_col = C_col_start + j;
                
                if (global_col < N) {
                    uint8_t b_mxfp8 = B_shared[k][local_col];
                    uint8_t b_scale = B_scales[scale_k_idx * N + global_col];
                    b_vals[j] = dequantize_mxfp8(b_mxfp8, b_scale);
                } else {
                    b_vals[j] = 0.0f;
                }
            }
            
            for (int i = 0; i < THREAD_TILE_M; i++) {
                for (int j = 0; j < THREAD_TILE_N; j++) {
                    C_reg[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    for (int i = 0; i < THREAD_TILE_M; i++) {
        for (int j = 0; j < THREAD_TILE_N; j++) {
            int global_row = C_row_start + i;
            int global_col = C_col_start + j;
            
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = C_reg[i][j];
            }
        }
    }
}

// Quantize matrix row-wise (for A)
void quantize_mxfp8_rowwise(const float* input, uint8_t* output, uint8_t* scales,
                             int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int block = 0; block < cols / 32; block++) {
            float max_val = 0.0f;
            for (int j = 0; j < 32; j++) {
                int idx = i * cols + block * 32 + j;
                max_val = fmaxf(max_val, fabsf(input[idx]));
            }
            
            int scale_exp = (max_val > 0) ? (int)ceilf(log2f(max_val)) : -127;
            uint8_t scale = scale_exp + 127;
            scales[i * (cols / 32) + block] = scale;
            
            float scale_val = powf(2.0f, (float)scale_exp);
            
            for (int j = 0; j < 32; j++) {
                int idx = i * cols + block * 32 + j;
                float val = input[idx];
                
                uint8_t sign = (val < 0) ? 1 : 0;
                float normalized = fabsf(val) / scale_val;
                
                int mantissa = (int)roundf((normalized - 1.0f) * 128.0f);
                mantissa = (mantissa < 0) ? 0 : (mantissa > 127 ? 127 : mantissa);
                
                output[idx] = (sign << 7) | mantissa;
            }
        }
    }
}

// Quantize matrix column-wise (for B)
void quantize_mxfp8_columnwise(const float* input, uint8_t* output, uint8_t* scales,
                                int rows, int cols) {
    // For each column
    for (int j = 0; j < cols; j++) {
        // For each 32-element block along the column
        for (int block = 0; block < rows / 32; block++) {
            float max_val = 0.0f;
            for (int i = 0; i < 32; i++) {
                int idx = (block * 32 + i) * cols + j;
                max_val = fmaxf(max_val, fabsf(input[idx]));
            }
            
            int scale_exp = (max_val > 0) ? (int)ceilf(log2f(max_val)) : -127;
            uint8_t scale = scale_exp + 127;
            scales[block * cols + j] = scale;  // B_scales[K/32, N]
            
            float scale_val = powf(2.0f, (float)scale_exp);
            
            for (int i = 0; i < 32; i++) {
                int idx = (block * 32 + i) * cols + j;
                float val = input[idx];
                
                uint8_t sign = (val < 0) ? 1 : 0;
                float normalized = fabsf(val) / scale_val;
                
                int mantissa = (int)roundf((normalized - 1.0f) * 128.0f);
                mantissa = (mantissa < 0) ? 0 : (mantissa > 127 ? 127 : mantissa);
                
                output[idx] = (sign << 7) | mantissa;
            }
        }
    }
}

int main() {
    int M = 128, N = 128, K = 128;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    
    uint8_t *h_A_mxfp8 = (uint8_t*)malloc(M * K);
    uint8_t *h_A_scales = (uint8_t*)malloc(M * (K / 32));
    uint8_t *h_B_mxfp8 = (uint8_t*)malloc(K * N);
    uint8_t *h_B_scales = (uint8_t*)malloc((K / 32) * N);
    
    // Initialize
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    
    // Quantize A row-wise, B column-wise
    quantize_mxfp8_rowwise(h_A, h_A_mxfp8, h_A_scales, M, K);
    quantize_mxfp8_columnwise(h_B, h_B_mxfp8, h_B_scales, K, N);
    
    // Device memory
    uint8_t *d_A_mxfp8, *d_A_scales, *d_B_mxfp8, *d_B_scales;
    float *d_C;
    
    cudaMalloc(&d_A_mxfp8, M * K);
    cudaMalloc(&d_A_scales, M * (K / 32));
    cudaMalloc(&d_B_mxfp8, K * N);
    cudaMalloc(&d_B_scales, (K / 32) * N);
    cudaMalloc(&d_C, size_C);
    
    cudaMemcpy(d_A_mxfp8, h_A_mxfp8, M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_scales, h_A_scales, M * (K / 32), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_mxfp8, h_B_mxfp8, K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_scales, h_B_scales, (K / 32) * N, cudaMemcpyHostToDevice);
    
    dim3 blockDim(16, 16);
    dim3 gridDim((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    
    printf("Launching kernel with grid(%d, %d) blocks(%d, %d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    fused_mxfp8_gemm_naive<<<gridDim, blockDim>>>(
        d_A_mxfp8, d_A_scales,
        d_B_mxfp8, d_B_scales,
        d_C, M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    printf("Success! Sample outputs:\n");
    printf("C[0][0] = %f\n", h_C[0]);
    printf("C[10][10] = %f\n", h_C[10 * N + 10]);
    printf("C[127][127] = %f\n", h_C[127 * N + 127]);
    
    // Cleanup
    free(h_A); free(h_B); free(h_C);
    free(h_A_mxfp8); free(h_A_scales);
    free(h_B_mxfp8); free(h_B_scales);
    cudaFree(d_A_mxfp8); cudaFree(d_A_scales);
    cudaFree(d_B_mxfp8); cudaFree(d_B_scales);
    cudaFree(d_C);
    
    return 0;
}