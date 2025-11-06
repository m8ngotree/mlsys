// B100

#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// Conservative parameters to avoid issues
const int BM = 64;   
const int BN = 64;   
const int BK = 64;    
const int TM = 8;    
const int TN = 8;    

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;
  const uint numThreadsBlocktile = (BM * BN) / (TM * TN);
  
  // Early exit if block is out of bounds
  if (cRow * BM >= M || cCol * BN >= K) return;
  
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);
  
  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  
  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint strideA = numThreadsBlocktile / BK;
  
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  const uint strideB = numThreadsBlocktile / BN;
  
  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN];
  for (int i = 0; i < TM * TN; i++) {
    threadResults[i] = 0.0f;
  }
  
  // register caches for As and Bs
  float regM[TM];
  float regN[TN];
  
  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < N; bkIdx += BK) {
    // populate the SMEM caches - Load As
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      int globalRowA = cRow * BM + innerRowA + loadOffset;
      int globalColA = bkIdx + innerColA;
      
      if (globalRowA < M && globalColA < N) {
        As[(innerRowA + loadOffset) * BK + innerColA] = A[globalRowA * N + globalColA];
      } else {
        As[(innerRowA + loadOffset) * BK + innerColA] = 0.0f;
      }
    }
    
    // populate the SMEM caches - Load Bs
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
    
    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // Load into registers with bounds checking
      for (uint i = 0; i < TM; ++i) {
        int sharedRowA = threadRow * TM + i;
        if (sharedRowA < BM && dotIdx < BK) {
          regM[i] = As[sharedRowA * BK + dotIdx];
        } else {
          regM[i] = 0.0f;
        }
      }
      
      for (uint i = 0; i < TN; ++i) {
        int sharedColB = threadCol * TN + i;
        if (dotIdx < BK && sharedColB < BN) {
          regN[i] = Bs[dotIdx * BN + sharedColB];
        } else {
          regN[i] = 0.0f;
        }
      }
      
      // Compute outer products
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }
  
  // write out the results with bounds checking
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
    dim3 threadsPerBlock((BM * BN) / (TM * TN));  // Should be 64 threads
    dim3 blocksPerGrid(CEIL_DIV(K, BN), CEIL_DIV(M, BM));
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}