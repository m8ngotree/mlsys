/**
 * Combined MXFP8 GEMM Kernels
 * 
 * This file combines all kernel variants into a single compilation unit.
 * Compile with: nvcc -O3 -arch=sm_80 -Xcompiler -fPIC -shared -o libmxfp8_gemm.so mxfp8_gemm_combined.cu
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>

using namespace nvcuda;

// ============================================================================
// Configuration
// ============================================================================

constexpr int MXFP_BLOCK_SIZE = 32;

// WMMA tile dimensions
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Block tile dimensions
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 32;

// ============================================================================
// E4M3 Dequantization LUT
// ============================================================================

__constant__ float g_e4m3_lut[256];

void init_e4m3_lut_internal() {
    float host_lut[256];
    
    for (int i = 0; i < 256; i++) {
        uint8_t bits = static_cast<uint8_t>(i);
        int sign = (bits >> 7) & 1;
        int exp = (bits >> 3) & 0xF;
        int mant = bits & 0x7;
        
        float value;
        if (exp == 0) {
            // Subnormal
            value = ldexpf(static_cast<float>(mant) / 8.0f, -6);
        } else if (exp == 15 && mant != 0) {
            // NaN
            value = nanf("");
        } else if (exp == 15 && mant == 0) {
            // Max value (no inf in E4M3)
            value = 448.0f;
        } else {
            // Normal
            value = ldexpf(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
        }
        
        host_lut[i] = sign ? -value : value;
    }
    
    cudaMemcpyToSymbol(g_e4m3_lut, host_lut, 256 * sizeof(float));
}

// ============================================================================
// Shared Memory Layout
// ============================================================================

struct GemmSharedMemory {
    __half A[2][BLOCK_M][BLOCK_K];
    __half B[2][BLOCK_K][BLOCK_N];
};

// ============================================================================
// Naive MXFP8 GEMM (Baseline)
// ============================================================================

__global__ void naive_mxfp8_gemm_kernel(
    const uint8_t* __restrict__ A_data,
    const float* __restrict__ A_scales,
    const uint8_t* __restrict__ B_data,
    const float* __restrict__ B_scales,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        for (int k = 0; k < K; k++) {
            uint8_t a_val = A_data[row * K + k];
            int a_scale_idx = row * (K / MXFP_BLOCK_SIZE) + k / MXFP_BLOCK_SIZE;
            float a = g_e4m3_lut[a_val] * A_scales[a_scale_idx];
            
            uint8_t b_val = B_data[k * N + col];
            int b_scale_idx = (k / MXFP_BLOCK_SIZE) * N + col;
            float b = g_e4m3_lut[b_val] * B_scales[b_scale_idx];
            
            sum += a * b;
        }
        
        C[row * N + col] = sum;
    }
}

// ============================================================================
// Tensor Core MXFP8 GEMM with Double Buffering
// ============================================================================

__global__ void tc_mxfp8_gemm_kernel(
    const uint8_t* __restrict__ A_data,
    const float* __restrict__ A_scales,
    const uint8_t* __restrict__ B_data,
    const float* __restrict__ B_scales,
    float* __restrict__ C,
    int M, int N, int K
) {
    extern __shared__ char smem_raw[];
    GemmSharedMemory* smem = reinterpret_cast<GemmSharedMemory*>(smem_raw);
    
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Warp tile assignment (2x4 warps handling 128x128 block)
    const int warp_row = (warp_id / 2) * 32;  // 0, 32, 64, 96
    const int warp_col = (warp_id % 2) * 64;  // 0, 64
    
    const int global_row = block_row * BLOCK_M;
    const int global_col = block_col * BLOCK_N;
    
    // Accumulator fragments: 2x4 WMMA tiles per warp
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][4];
    
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_a[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_b[4];
    
    int buf = 0;
    const int load_stride = blockDim.x;
    
    // Load and dequantize first A tile
    for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_K; idx += load_stride) {
        int row = idx / BLOCK_K;
        int col = idx % BLOCK_K;
        int g_row = global_row + row;
        int g_col = col;
        
        if (g_row < M && g_col < K) {
            uint8_t val = A_data[g_row * K + g_col];
            int scale_idx = g_row * (K / MXFP_BLOCK_SIZE) + g_col / MXFP_BLOCK_SIZE;
            float scale = A_scales[scale_idx];
            smem->A[buf][row][col] = __float2half(g_e4m3_lut[val] * scale);
        } else {
            smem->A[buf][row][col] = __float2half(0.0f);
        }
    }
    
    // Load and dequantize first B tile
    for (int idx = threadIdx.x; idx < BLOCK_K * BLOCK_N; idx += load_stride) {
        int row = idx / BLOCK_N;
        int col = idx % BLOCK_N;
        int g_row = row;
        int g_col = global_col + col;
        
        if (g_row < K && g_col < N) {
            uint8_t val = B_data[g_row * N + g_col];
            int scale_idx = (g_row / MXFP_BLOCK_SIZE) * N + g_col;
            float scale = B_scales[scale_idx];
            smem->B[buf][row][col] = __float2half(g_e4m3_lut[val] * scale);
        } else {
            smem->B[buf][row][col] = __float2half(0.0f);
        }
    }
    
    __syncthreads();
    
    const int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;
    
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int next_buf = 1 - buf;
        int next_k = (k_tile + 1) * BLOCK_K;
        
        // Prefetch next tiles
        if (k_tile < num_k_tiles - 1) {
            for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_K; idx += load_stride) {
                int row = idx / BLOCK_K;
                int col = idx % BLOCK_K;
                int g_row = global_row + row;
                int g_col = next_k + col;
                
                if (g_row < M && g_col < K) {
                    uint8_t val = A_data[g_row * K + g_col];
                    int scale_idx = g_row * (K / MXFP_BLOCK_SIZE) + g_col / MXFP_BLOCK_SIZE;
                    float scale = A_scales[scale_idx];
                    smem->A[next_buf][row][col] = __float2half(g_e4m3_lut[val] * scale);
                } else {
                    smem->A[next_buf][row][col] = __float2half(0.0f);
                }
            }
            
            for (int idx = threadIdx.x; idx < BLOCK_K * BLOCK_N; idx += load_stride) {
                int row = idx / BLOCK_N;
                int col = idx % BLOCK_N;
                int g_row = next_k + row;
                int g_col = global_col + col;
                
                if (g_row < K && g_col < N) {
                    uint8_t val = B_data[g_row * N + g_col];
                    int scale_idx = (g_row / MXFP_BLOCK_SIZE) * N + g_col;
                    float scale = B_scales[scale_idx];
                    smem->B[next_buf][row][col] = __float2half(g_e4m3_lut[val] * scale);
                } else {
                    smem->B[next_buf][row][col] = __float2half(0.0f);
                }
            }
        }
        
        // Compute on current buffer
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k += WMMA_K) {
            #pragma unroll
            for (int m = 0; m < 2; m++) {
                int a_row = warp_row + m * WMMA_M;
                wmma::load_matrix_sync(frag_a[m], &smem->A[buf][a_row][k], BLOCK_K);
            }
            
            #pragma unroll
            for (int n = 0; n < 4; n++) {
                int b_col = warp_col + n * WMMA_N;
                wmma::load_matrix_sync(frag_b[n], &smem->B[buf][k][b_col], BLOCK_N);
            }
            
            #pragma unroll
            for (int m = 0; m < 2; m++) {
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    wmma::mma_sync(acc[m][n], frag_a[m], frag_b[n], acc[m][n]);
                }
            }
        }
        
        buf = next_buf;
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int m = 0; m < 2; m++) {
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int c_row = global_row + warp_row + m * WMMA_M;
            int c_col = global_col + warp_col + n * WMMA_N;
            
            if (c_row < M && c_col < N) {
                wmma::store_matrix_sync(&C[c_row * N + c_col], acc[m][n], N, wmma::mem_row_major);
            }
        }
    }
}

// ============================================================================
// Host Interface
// ============================================================================

extern "C" {

void init_luts() {
    init_e4m3_lut_internal();
}

void naive_mxfp8_gemm(
    const uint8_t* A_data,
    const float* A_scales,
    const uint8_t* B_data,
    const float* B_scales,
    float* C,
    int M, int N, int K
) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    naive_mxfp8_gemm_kernel<<<grid, block>>>(
        A_data, A_scales, B_data, B_scales, C, M, N, K
    );
}

void mxfp8_tc_gemm(
    const uint8_t* A_data,
    const float* A_scales,
    const uint8_t* B_data,
    const float* B_scales,
    float* C,
    int M, int N, int K
) {
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(256);
    
    size_t smem_size = sizeof(GemmSharedMemory);
    
    tc_mxfp8_gemm_kernel<<<grid, block, smem_size>>>(
        A_data, A_scales, B_data, B_scales, C, M, N, K
    );
}

}  // extern "C"
