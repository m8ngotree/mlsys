#!/usr/bin/env python3
"""
Verification script to compare CUDA quantization kernels with PyTorch reference.
"""

import torch
import numpy as np
import subprocess
import os
import tempfile
import re
from pathlib import Path

def generate_test_data(N, seed=42):
    """Generate test data matching PyTorch quantization."""
    torch.manual_seed(seed)
    x = torch.randn(N, dtype=torch.float16, device='cuda') * 2.0
    return x

def pytorch_quantize(x):
    """PyTorch reference quantization."""
    scale = x.abs().max() / 127.0
    x_quant = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return x_quant.cpu().numpy(), scale.item()

def save_data_to_file(data, filename):
    """Save torch tensor to binary file."""
    data_cpu = data.cpu().numpy()
    # Write as half precision (2 bytes per element)
    # Use numpy's tobytes which writes in native byte order
    with open(filename, 'wb') as f:
        f.write(data_cpu.astype(np.float16).tobytes())

def load_int8_from_file(filename, N):
    """Load int8 data from binary file."""
    with open(filename, 'rb') as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.int8)

def extract_kernel_code(kernel_file):
    """Extract kernel code from CUDA file, excluding main()."""
    with open(kernel_file, 'r') as f:
        content = f.read()
    
    # Find the main function and everything before it
    main_match = re.search(r'\s*int\s+main\s*\(', content)
    if main_match:
        kernel_code = content[:main_match.start()]
    else:
        kernel_code = content
    
    return kernel_code

def create_test_harness(kernel_code, kernel_file):
    """Create a test harness that uses the extracted kernel code."""
    kernel_name = Path(kernel_file).stem
    
    # Check if vectorized kernel exists by looking for the function name in kernel code
    has_vec4 = "quantize_kernel_vec4" in kernel_code
    
    harness = f"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <iostream>

// Kernel code from {kernel_file}
{kernel_code}

int main(int argc, char* argv[]) {{
    if (argc != 4) {{
        printf("Usage: %s <input_file> <output_file> <N>\\n", argv[0]);
        return 1;
    }}
    
    const int N = atoi(argv[3]);
    const int bytes_fp16 = N * sizeof(half);
    const int bytes_int8 = N * sizeof(int8_t);
    
    // Read input from file
    half* h_input = (half*)malloc(bytes_fp16);
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) {{
        printf("Error: Cannot open input file\\n");
        return 1;
    }}
    fread(h_input, sizeof(half), N, fin);
    fclose(fin);
    
    half *d_input;
    int8_t *d_quantized;
    float *d_max_val;
    
    cudaMalloc(&d_input, bytes_fp16);
    cudaMalloc(&d_quantized, bytes_int8);
    cudaMalloc(&d_max_val, sizeof(float));
    
    float init_val = 0.0f;
    cudaMemcpy(d_max_val, &init_val, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input, h_input, bytes_fp16, cudaMemcpyHostToDevice);
    
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    
    // Find max
    find_abs_max_kernel<<<num_blocks, block_size>>>(d_input, d_max_val, N);
    cudaDeviceSynchronize();
    
    float max_val;
    cudaMemcpy(&max_val, d_max_val, sizeof(float), cudaMemcpyDeviceToHost);
    float scale = max_val / 127.0f;
    
    // Quantize
    cudaError_t err;
    """
    
    if has_vec4:
        harness += """
    // Use vectorized kernel
    int num_blocks_vec4 = (N + (block_size * 4) - 1) / (block_size * 4);
    quantize_kernel_vec4<<<num_blocks_vec4, block_size>>>(d_input, d_quantized, scale, N);
    """
    else:
        harness += """
    // Use regular kernel
    quantize_kernel<<<num_blocks, block_size>>>(d_input, d_quantized, scale, N);
    """
    
    harness += """
    cudaDeviceSynchronize();
    
    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Write output
    int8_t* h_quantized = (int8_t*)malloc(bytes_int8);
    cudaMemcpy(h_quantized, d_quantized, bytes_int8, cudaMemcpyDeviceToHost);
    
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) {
        printf("Error: Cannot open output file\\n");
        return 1;
    }
    fwrite(h_quantized, sizeof(int8_t), N, fout);
    fclose(fout);
    
    // Print scale to stdout for verification
    printf("SCALE:%f\\n", scale);
    
    free(h_input);
    free(h_quantized);
    cudaFree(d_input);
    cudaFree(d_quantized);
    cudaFree(d_max_val);
    
    return 0;
}
"""
    return harness

def compile_and_run_kernel(kernel_file, input_file, output_file, N):
    """Compile and run a CUDA kernel, returning the quantized output and scale."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Extract kernel code
        kernel_code = extract_kernel_code(kernel_file)
        
        # Create test harness
        harness_code = create_test_harness(kernel_code, kernel_file)
        
        harness_file = os.path.join(temp_dir, "test_harness.cu")
        with open(harness_file, 'w') as f:
            f.write(harness_code)
        
        exe_file = os.path.join(temp_dir, "test_kernel")
        
        # Compile
        compile_cmd = ["nvcc", harness_file, "-o", exe_file, "-std=c++11"]
        try:
            result = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"  Compilation failed:")
            print(f"  {e.stderr}")
            return None, None
        
        # Run
        run_cmd = [exe_file, input_file, output_file, str(N)]
        try:
            result = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            # Extract scale from stdout
            scale = None
            for line in result.stdout.split('\n'):
                if line.startswith('SCALE:'):
                    scale = float(line.split(':')[1])
            
            output_data = load_int8_from_file(output_file, N)
            return output_data, scale
        except subprocess.CalledProcessError as e:
            print(f"  Execution failed:")
            print(f"  {e.stderr}")
            if e.stdout:
                print(f"  stdout: {e.stdout}")
            return None, None
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def compare_outputs(ref_output, kernel_output, kernel_name, tolerance=0):
    """Compare kernel output with reference."""
    if kernel_output is None:
        print(f"  ❌ {kernel_name}: Failed to run")
        return False
    
    if ref_output.shape != kernel_output.shape:
        print(f"  ❌ {kernel_name}: Shape mismatch (ref: {ref_output.shape}, kernel: {kernel_output.shape})")
        return False
    
    # Check for exact match (quantization should be deterministic)
    matches = np.array_equal(ref_output, kernel_output)
    
    if matches:
        print(f"  ✅ {kernel_name}: Output matches PyTorch exactly")
        return True
    else:
        # Count differences
        diff_mask = ref_output != kernel_output
        num_diffs = np.sum(diff_mask)
        max_diff = np.max(np.abs(ref_output.astype(np.int16) - kernel_output.astype(np.int16)))
        
        print(f"  ❌ {kernel_name}: {num_diffs}/{len(ref_output)} values differ (max diff: {max_diff})")
        
        # Show first few differences
        if num_diffs > 0:
            diff_indices = np.where(diff_mask)[0][:5]
            print(f"     First differences:")
            for idx in diff_indices:
                print(f"       [{idx}]: ref={ref_output[idx]}, kernel={kernel_output[idx]}")
        
        return False

def verify_kernels(N=1024*1024, test_cases=None):
    """Verify all quantization kernels against PyTorch."""
    print(f"Generating test data (N={N})...")
    x = generate_test_data(N)
    
    print("Computing PyTorch reference...")
    ref_output, ref_scale = pytorch_quantize(x)
    
    print(f"Reference scale: {ref_scale:.6f}")
    print(f"Reference output range: [{ref_output.min()}, {ref_output.max()}]")
    print()
    
    # Get all kernel files
    kernel_dir = Path(__file__).parent
    kernel_files = sorted(kernel_dir.glob("quantize_*.cu"))
    
    if test_cases:
        kernel_files = [f for f in kernel_files if any(tc in f.name for tc in test_cases)]
    
    print(f"Testing {len(kernel_files)} kernel(s)...")
    print()
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as input_file:
        input_path = input_file.name
        save_data_to_file(x, input_path)
    
    results = {}
    
    try:
        for kernel_file in kernel_files:
            kernel_name = kernel_file.stem
            print(f"Testing {kernel_name}...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as output_file:
                output_path = output_file.name
            
            kernel_output, kernel_scale = compile_and_run_kernel(
                str(kernel_file),
                input_path,
                output_path,
                N
            )
            
            if kernel_scale is not None:
                scale_diff = abs(kernel_scale - ref_scale)
                if scale_diff > 1e-5:
                    print(f"  ⚠️  Scale difference: {scale_diff:.6f} (ref: {ref_scale:.6f}, kernel: {kernel_scale:.6f})")
            
            is_correct = compare_outputs(ref_output, kernel_output, kernel_name)
            results[kernel_name] = is_correct
            
            # Cleanup output file
            os.unlink(output_path)
            print()
    
    finally:
        # Cleanup input file
        os.unlink(input_path)
    
    # Summary
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    all_passed = all(results.values())
    for kernel_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {kernel_name}")
    
    if all_passed:
        print("\n🎉 All kernels match PyTorch reference!")
    else:
        print("\n⚠️  Some kernels have mismatches with PyTorch reference.")
    
    return all_passed

if __name__ == "__main__":
    import sys
    
    # Default test size
    N = 1024 * 1024
    
    # Allow specifying test cases
    test_cases = None
    if len(sys.argv) > 1:
        if sys.argv[1].isdigit():
            N = int(sys.argv[1])
            if len(sys.argv) > 2:
                test_cases = sys.argv[2:]
        else:
            test_cases = sys.argv[1:]
    
    success = verify_kernels(N=N, test_cases=test_cases)
    sys.exit(0 if success else 1)
