#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

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

// Kernel 1: Find max absolute value (for scale computation)
__global__ void find_abs_max_kernel(const half* input, float* max_val, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_max = 0.0f;

    if (idx < N) {
        local_max = fabsf(__half2float(input[idx]));
    }

    if (local_max > 0.0f) {
        atomicMaxFloat(max_val, local_max);
    }
}

// Kernel 2: Quantize using computed scale
__global__ void quantize_kernel(const half* input, int8_t* output, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float val = __half2float(input[idx]);
        float quantized = roundf(val / scale);

        // Clamp to INT8 range [-127, 127]
        quantized = fmaxf(-127.0f, fminf(127.0f, quantized));
        output[idx] = (int8_t)quantized;
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

    find_abs_max_kernel<<<num_blocks, block_size>>>(d_input, d_max_val, N);
    cudaDeviceSynchronize();

    float max_val;
    cudaMemcpy(&max_val, d_max_val, sizeof(float), cudaMemcpyDeviceToHost);
    float scale = max_val / 127.0f;
    
    printf("Max absolute value %f\n", max_val);
    printf("Scale %f\n", scale);

    quantize_kernel<<<num_blocks, block_size>>>(d_input, d_quantized, scale, N);

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
    
    return 0;
}