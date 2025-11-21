// https://maharshi.bearblog.dev/optimizing-softmax-cuda/

#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define CUDA_CHECK(ans)                        \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}
#define CEIL_DIV(x, y) ((x) >= 0 ? (((x) + (y) - 1) / (y)) : ((x) / (y)))

#define M_PI 3.14159265f

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif


template<typename T, typename Op>
__device__ __forceinline__ T warpReduce(T val, Op op, unsigned int mask = 0xffffffffu) {
    for(int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
        val = op(val, __shfl_down_sync(mask, val, offset));
    return val;
}

template<typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    return warpReduce(val, []__device__(T a, T b) { return a + b; });
}

template<typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
    return warpReduce(val, []__device__(T a, T b) { return a > b ? a : b; });
}

template<typename T, typename Op>
__device__ __forceinline__ void blockReduce(T val, T *smem, T identity, Op op) {
    int tx = threadIdx.x;
    int wid = tx / WARP_SIZE;
    int lane = tx % WARP_SIZE;

    val = warpReduce(val, op);

    if (blockDim.x > WARP_SIZE) {
        if (lane == 0) {
            smem[wid] = val;
        }
        __syncthreads();

        if (tx < WARP_SIZE) {
            val = (tx < CEIL_DIV(blockDim.x, WARP_SIZE)) ? smem[tx] : identity;
            val = warpReduce(val, op);
            if (tx == 0) smem[0] = val;
        }
    } else {
        if (tx == 0) smem[0] = val;
    }
}

template<typename T>
__device__ __forceinline__ void blockReduceSum(T val, T *smem, T identity) {
    return blockReduce(
        val, smem, identity, []__device__(T a, T b) { return a + b; } 
    );
}


template<typename T>
__device__ __forceinline__ void blockReduceMax(T val, T *smem, T identity) {
    return blockReduce(
        val, smem, identity, []__device__(T a, T b) { return a > b ? a : b; }
    );
}

#endif  // CUDA_UTILS_CUH

