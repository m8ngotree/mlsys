// https://siboehm.com/articles/22/CUDA-MMM

#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    extern __shared__ float shared_mem[];
    
    float* Mds = shared_mem;
    float* Nds = shared_mem + TILE_WIDTH * TILE_WIDTH;
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0.0f;

    for (int ph = 0; ph < (N + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {
        
        if ((Row < M) && ((ph * TILE_WIDTH + tx) < N)) {
            Mds[ty * TILE_WIDTH + tx] = A[Row * N + ph * TILE_WIDTH + tx];
        } else {
            Mds[ty * TILE_WIDTH + tx] = 0.0f;
        }
        
        if ((ph * TILE_WIDTH + ty) < N && (Col < K)) {
            Nds[ty * TILE_WIDTH + tx] = B[(ph * TILE_WIDTH + ty) * K + Col];
        } else {
            Nds[ty * TILE_WIDTH + tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += Mds[ty * TILE_WIDTH + k] * Nds[k * TILE_WIDTH + tx];
        }
        
        __syncthreads();
    }

    if ((Row < M) && (Col < K)) {
        C[Row * K + Col] = Pvalue;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    size_t shared_mem_size = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
