#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0.0f;
    
    for (int ph = 0; ph < (N + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        if (Row < M && (ph * TILE_WIDTH + tx) < N) {
            Mds[ty][tx] = A[Row * N + ph * TILE_WIDTH + tx];
        } else {
            Mds[ty][tx] = 0.0f;
        }
        
        if ((ph * TILE_WIDTH + ty) < N && Col < K) {
            Nds[ty][tx] = B[(ph * TILE_WIDTH + ty) * K + Col];
        } else {
            Nds[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        
        __syncthreads();
    }
    
    if (Row < M && Col < K) {
        C[Row * K + Col] = Pvalue;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
