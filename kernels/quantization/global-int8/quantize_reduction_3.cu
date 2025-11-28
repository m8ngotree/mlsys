#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

// Define half4 struct (CUDA doesn't have a built-in half4 type)
struct half4 {
    half x, y, z, w;
};

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int expected;
    
    do {
        expected = old;
        old = atomicCAS(addr_as_int, expected, __float_as_int(fmaxf(value, __int_as_float(expected))));
    } while (expected != old);
    
    return __int_as_float(old);
}

__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Kernel 1: Find max absolute value with warp shuffle + minimal shared memory
__global__ void find_abs_max_kernel_warp(const half* input, float* max_val, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_max = 0.0f;
    if (idx < N) {
        local_max = fabsf(__half2float(input[idx]));
    }
    
    local_max = warpReduceMax(local_max);
    
    __shared__ float warp_maxes[8];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (lane_id == 0) {
        warp_maxes[warp_id] = local_max;
    }
    __syncthreads();
    
    if (tid < 32) {
        local_max = (tid < 8) ? warp_maxes[tid] : -INFINITY;
        
        local_max = warpReduceMax(local_max);
        
        if (tid == 0 && local_max > 0.0f) {
            atomicMaxFloat(max_val, local_max);
        }
    }
}

// Kernel 2: Vectorized Quantize (4 elements per thread)
__global__ void quantize_kernel_vec4(const half* input, int8_t* output, float scale, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < N) {
        half4 in_vals = *((half4*)&input[idx]);

        char4 out_vals;
        out_vals.x = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(__half2float(in_vals.x) / scale)));
        out_vals.y = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(__half2float(in_vals.y) / scale)));
        out_vals.z = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(__half2float(in_vals.z) / scale)));
        out_vals.w = (int8_t)fmaxf(-127.0f, fminf(127.0f, roundf(__half2float(in_vals.w) / scale)));

        *((char4*)&output[idx]) = out_vals;
    } else if (idx < N) {
        for (int i = idx; i < N && i < idx + 4; i++) {
            float val = __half2float(input[i]);
            float quantized = roundf(val / scale);
            quantized = fmaxf(-127.0f, fminf(127.0f, quantized));
            output[i] = (int8_t)quantized;
        }
    }
}

// Kernel 3: Dequantize
__global__ void dequantize_kernel(const int8_t* input, half* output, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float val = (float)input[idx] * scale;
        output[idx] = __float2half(val);
    }
}

int main() {
    const int N = 1024 * 1024;
    const int bytes_fp16 = N * sizeof(half);
    const int bytes_int8 = N * sizeof(int8_t);

    half* h_input = (half*)malloc(bytes_fp16);
    int8_t* h_quantized = (int8_t*)malloc(bytes_int8);
    half* h_dequantized = (half*)malloc(bytes_fp16);

    // Initialize with random values in range [-2, 2]
    srand(42);
    for (int i = 0; i < N; i++) {
        float val = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
        h_input[i] = __float2half(val);
    }

    half *d_input, *d_dequantized;
    int8_t *d_quantized;
    float *d_max_val;

    cudaMalloc(&d_input, bytes_fp16);
    cudaMalloc(&d_quantized, bytes_int8);
    cudaMalloc(&d_dequantized, bytes_fp16);
    cudaMalloc(&d_max_val, sizeof(float));

    float init_val = 0.0f;
    cudaMemcpy(d_max_val, &init_val, sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_input, h_input, bytes_fp16, cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;

    find_abs_max_kernel_warp<<<num_blocks, block_size>>>(d_input, d_max_val, N);
    cudaDeviceSynchronize();

    float max_val;
    cudaMemcpy(&max_val, d_max_val, sizeof(float), cudaMemcpyDeviceToHost);
    float scale = max_val / 127.0f;
    
    printf("Max absolute value %f\n", max_val);
    printf("Scale %f\n", scale);

    int num_blocks_vec4 = (N + (block_size * 4) - 1) / (block_size * 4);
    quantize_kernel_vec4<<<num_blocks_vec4, block_size>>>(d_input, d_quantized, scale, N);

    dequantize_kernel<<<num_blocks, block_size>>>(d_quantized, d_dequantized, scale, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_quantized, d_quantized, bytes_int8, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dequantized, d_dequantized, bytes_fp16, cudaMemcpyDeviceToHost);

    float max_error = 0.0f;
    float avg_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float original = __half2float(h_input[i]);
        float reconstructed = __half2float(h_dequantized[i]);
        float error = fabsf(original - reconstructed);
        max_error = std::max(max_error, error);
        avg_error += error;
    }
    avg_error /= N;

    printf("\n=== Validation Results ===\n");
    printf("Max quantization error: %f\n", max_error);
    printf("Average quantization error: %f\n", avg_error);
    printf("Expected max error (scale): %f\n", scale);

    printf("\n=== Sample Values ===\n");
    for (int i = 0; i < 5; i++) {
        printf("Original: %7.4f -> Quantized: %4d -> Dequantized: %7.4f (error: %.4f)\n",
               __half2float(h_input[i]),
               h_quantized[i],
               __half2float(h_dequantized[i]),
               fabsf(__half2float(h_input[i]) - __half2float(h_dequantized[i])));
    }

    free(h_input);
    free(h_quantized);
    free(h_dequantized);
    cudaFree(d_input);
    cudaFree(d_quantized);
    cudaFree(d_dequantized);
    cudaFree(d_max_val);

    // Benchmark
    const int bench_N = 128 * 1024 * 1024; // 128M elements (256 MB FP16)
    const int bench_bytes_fp16 = bench_N * sizeof(half);
    const int bench_bytes_int8 = bench_N * sizeof(int8_t);

    half *d_bench_input;
    int8_t *d_bench_output;
    cudaMalloc(&d_bench_input, bench_bytes_fp16);
    cudaMalloc(&d_bench_output, bench_bytes_int8);

    int bench_blocks_vec4 = (bench_N + (block_size * 4) - 1) / (block_size * 4);

    // Warmup
    for (int i = 0; i < 10; i++) {
        quantize_kernel_vec4<<<bench_blocks_vec4, block_size>>>(d_bench_input, d_bench_output, scale, bench_N);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iters = 100;

    cudaEventRecord(start);
    for (int i = 0; i < num_iters; i++) {
        quantize_kernel_vec4<<<bench_blocks_vec4, block_size>>>(d_bench_input, d_bench_output, scale, bench_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / num_iters;
    float bytes_transferred = bench_bytes_fp16 + bench_bytes_int8; // read FP16 + write INT8
    float bandwidth_gbs = (bytes_transferred / avg_ms) / 1e6; // GB/s

    printf("\n=== Benchmark Results ===\n");
    printf("Problem size: %d elements (%.2f MB)\n", bench_N, bytes_transferred / 1e6);
    printf("Quantization: %.3f ms, %.2f GB/s\n", avg_ms, bandwidth_gbs);

    cudaFree(d_bench_input);
    cudaFree(d_bench_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}