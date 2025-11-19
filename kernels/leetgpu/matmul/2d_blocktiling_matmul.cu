// https://siboehm.com/articles/22/CUDA-MMM

#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

const int BM = 64;   
const int BN = 64;   
const int BK = 8;
const int TM = 8;    
const int TN = 8;    

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;
  
  if (cRow * BM >= M || cCol * BN >= K) return;
  
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);
  
  __shared__ float As[BK * BM];
  __shared__ float Bs[BK * BN];
  
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint strideA = blockDim.x / BK;
  
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  const uint strideB = blockDim.x / BN;
  
  float threadResults[TM * TN];
  for (int i = 0; i < TM * TN; i++) {
    threadResults[i] = 0.0f;
  }
  
  float regM[TM];
  float regN[TN];
  
  for (uint bkIdx = 0; bkIdx < N; bkIdx += BK) {
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      int globalRowA = cRow * BM + innerRowA + loadOffset;
      int globalColA = bkIdx + innerColA;
      
      if (globalRowA < M && globalColA < N) {
        As[innerColA * BM + (innerRowA + loadOffset)] = A[globalRowA * N + globalColA];
      } else {
        As[innerColA * BM + (innerRowA + loadOffset)] = 0.0f;
      }
    }
    
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      int globalRowB = bkIdx + innerRowB + loadOffset;
      int globalColB = cCol * BN + innerColB;
      
      if (globalRowB < N && globalColB < K) {
        Bs[(innerRowB + loadOffset) * BN + innerColB] = B[globalRowB * K + globalColB];
      } else {
        Bs[(innerRowB + loadOffset) * BN + innerColB] = 0.0f;
      }
    }
    __syncthreads();
    
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      #pragma unroll
      for (uint i = 0; i < TM; ++i) {
        int sharedRowA = threadRow * TM + i;
        regM[i] = As[dotIdx * BM + sharedRowA];
      }
      
      #pragma unroll
      for (uint i = 0; i < TN; ++i) {
        int sharedColB = threadCol * TN + i;
        regN[i] = Bs[dotIdx * BN + sharedColB];
      }
      
      #pragma unroll
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        #pragma unroll
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }
  
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      int globalRow = cRow * BM + threadRow * TM + resIdxM;
      int globalCol = cCol * BN + threadCol * TN + resIdxN;
      
      if (globalRow < M && globalCol < K) {
        C[globalRow * K + globalCol] = threadResults[resIdxM * TN + resIdxN];
      }
    }
  }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock((BM * BN) / (TM * TN));
    dim3 blocksPerGrid(CEIL_DIV(K, BN), CEIL_DIV(M, BM));
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}