#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

__global__ void pack_int4_naive(
    const int8_t* input,
    uint8_t* output,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int input_idx = idx * 2;
    
    if (input_idx + 1 < N) {
        int8_t val0 = input[input_idx];
        int8_t val1 = input[input_idx + 1];
        
        uint8_t nibble0 = val0 & 0x0F;
        uint8_t nibble1 = val1 & 0x0F;
        
        uint8_t packed = (nibble0 << 4) | nibble1;
        
        output[idx] = packed;
    }
    else if (input_idx < N) {
        int8_t val0 = input[input_idx];
        uint8_t nibble0 = val0 & 0x0F;
        output[idx] = nibble0 << 4;
    }
}

__global__ void unpack_int4_naive(
    const uint8_t* input,
    int8_t* output,
    int packed_N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < packed_N) {
        uint8_t packed = input[idx];
        
        uint8_t high_nibble = packed >> 4;
        uint8_t low_nibble = packed & 0x0F;
        
        int8_t val0 = (int8_t)(high_nibble << 4) >> 4;
        int8_t val1 = (int8_t)(low_nibble << 4) >> 4;
        
        output[idx * 2] = val0;
        output[idx * 2 + 1] = val1;
    }
}

int main() {
    const int N = 1024 * 1024;
    const int packed_N = (N + 1) / 2;
    
    const int bytes_unpacked = N * sizeof(int8_t);
    const int bytes_packed = packed_N * sizeof(uint8_t);

    int8_t* h_input = (int8_t*)malloc(bytes_unpacked);
    uint8_t* h_packed = (uint8_t*)malloc(bytes_packed);
    int8_t* h_unpacked = (int8_t*)malloc(bytes_unpacked);

    srand(42);
    for (int i = 0; i < N; i++) {
        int val = (rand() % 15) - 7;
        h_input[i] = (int8_t)val;
    }

    int8_t *d_input, *d_unpacked;
    uint8_t *d_packed;

    cudaMalloc(&d_input, bytes_unpacked);
    cudaMalloc(&d_packed, bytes_packed);
    cudaMalloc(&d_unpacked, bytes_unpacked);
    
    cudaMemcpy(d_input, h_input, bytes_unpacked, cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks_pack = (packed_N + block_size - 1) / block_size;

    pack_int4_naive<<<num_blocks_pack, block_size>>>(d_input, d_packed, N);
    cudaDeviceSynchronize();

    unpack_int4_naive<<<num_blocks_pack, block_size>>>(d_packed, d_unpacked, packed_N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_packed, d_packed, bytes_packed, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_unpacked, d_unpacked, bytes_unpacked, cudaMemcpyDeviceToHost);

    int errors = 0;
    int first_error_idx = -1;
    int8_t first_error_val = 0;
    
    for (int i = 0; i < N; i++) {
        if (h_input[i] != h_unpacked[i]) {
            errors++;
            if (errors == 1) {
                first_error_idx = i;
                first_error_val = h_unpacked[i];
            }
        }
    }

    printf("\n=== Validation Results ===\n");
    printf("Total elements: %d\n", N);
    printf("Packed bytes: %d\n", packed_N);
    printf("Compression ratio: %.2fx\n", (float)N / packed_N);
    printf("Errors: %d\n", errors);
    
    if (errors > 0) {
        printf("First error at index %d: expected %d, got %d\n", 
               first_error_idx, h_input[first_error_idx], first_error_val);
    } else {
        printf("✓ All values match correctly!\n");
    }

    printf("\n=== Sample Values ===\n");
    for (int i = 0; i < 10 && i < N; i++) {
        printf("Index %4d: Original=%4d -> Packed[%d]=0x%02X -> Unpacked=%4d %s\n",
               i, h_input[i], i/2, h_packed[i/2], h_unpacked[i],
               (h_input[i] == h_unpacked[i]) ? "✓" : "✗");
    }

    printf("\n=== Packing Details (First 5 elements) ===\n");
    for (int i = 0; i < 5 && i < N; i += 2) {
        if (i + 1 < N) {
            printf("Elements [%d,%d]: %d, %d -> 0x%02X\n",
                   i, i+1, h_input[i], h_input[i+1], h_packed[i/2]);
        } else {
            printf("Element [%d]: %d -> 0x%02X (padded)\n",
                   i, h_input[i], h_packed[i/2]);
        }
    }

    free(h_input);
    free(h_packed);
    free(h_unpacked);
    cudaFree(d_input);
    cudaFree(d_packed);
    cudaFree(d_unpacked);

    const int bench_N = 128 * 1024 * 1024;
    const int bench_packed_N = (bench_N + 1) / 2;
    const int bench_bytes_unpacked = bench_N * sizeof(int8_t);
    const int bench_bytes_packed = bench_packed_N * sizeof(uint8_t);

    int8_t *d_bench_input;
    uint8_t *d_bench_packed;
    int8_t *d_bench_unpacked;
    
    cudaMalloc(&d_bench_input, bench_bytes_unpacked);
    cudaMalloc(&d_bench_packed, bench_bytes_packed);
    cudaMalloc(&d_bench_unpacked, bench_bytes_unpacked);

    int bench_blocks = (bench_packed_N + block_size - 1) / block_size;

    for (int i = 0; i < 10; i++) {
        pack_int4_naive<<<bench_blocks, block_size>>>(d_bench_input, d_bench_packed, bench_N);
        unpack_int4_naive<<<bench_blocks, block_size>>>(d_bench_packed, d_bench_unpacked, bench_packed_N);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_iters = 100;

    cudaEventRecord(start);
    for (int i = 0; i < num_iters; i++) {
        pack_int4_naive<<<bench_blocks, block_size>>>(d_bench_input, d_bench_packed, bench_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms_pack = ms / num_iters;
    float bytes_transferred_pack = bench_bytes_unpacked + bench_bytes_packed;
    float bandwidth_gbs_pack = (bytes_transferred_pack / avg_ms_pack) / 1e6;

    cudaEventRecord(start);
    for (int i = 0; i < num_iters; i++) {
        unpack_int4_naive<<<bench_blocks, block_size>>>(d_bench_packed, d_bench_unpacked, bench_packed_N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms_unpack = ms / num_iters;
    float bytes_transferred_unpack = bench_bytes_packed + bench_bytes_unpacked;
    float bandwidth_gbs_unpack = (bytes_transferred_unpack / avg_ms_unpack) / 1e6;

    printf("\n=== Benchmark Results ===\n");
    printf("Problem size: %d INT4 values (%.2f MB unpacked, %.2f MB packed)\n", 
           bench_N, bench_bytes_unpacked / 1e6, bench_bytes_packed / 1e6);
    printf("Packing:   %.3f ms, %.2f GB/s\n", avg_ms_pack, bandwidth_gbs_pack);
    printf("Unpacking: %.3f ms, %.2f GB/s\n", avg_ms_unpack, bandwidth_gbs_unpack);

    cudaFree(d_bench_input);
    cudaFree(d_bench_packed);
    cudaFree(d_bench_unpacked);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

