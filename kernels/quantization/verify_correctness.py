#!/usr/bin/env python3
"""
Claude-generated verification script to compare CUDA quantization kernels with PyTorch reference.
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
    
    # Detect which max-finding kernel function is available
    max_kernel_name = None
    if "find_abs_max_kernel_warp" in kernel_code:
        max_kernel_name = "find_abs_max_kernel_warp"
    elif "find_abs_max_kernel_optimized" in kernel_code:
        max_kernel_name = "find_abs_max_kernel_optimized"
    elif "find_abs_max_kernel" in kernel_code:
        max_kernel_name = "find_abs_max_kernel"
    else:
        raise ValueError(f"Could not find max-finding kernel in {kernel_file}")
    
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
    
    // Find max using the detected kernel function
    {max_kernel_name}<<<num_blocks, block_size>>>(d_input, d_max_val, N);
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

def compare_outputs(ref_output, kernel_output, kernel_name, ref_scale=None, kernel_scale=None, tolerance=1):
    """Compare kernel output with reference.
    
    Args:
        ref_output: Reference quantized output from PyTorch
        kernel_output: Quantized output from CUDA kernel
        kernel_name: Name of the kernel being tested
        ref_scale: Reference scale from PyTorch
        kernel_scale: Scale from CUDA kernel
        tolerance: Maximum allowed difference in quantized values (default: 1, allows ±1 differences)
    """
    if kernel_output is None:
        print(f"  [FAIL] {kernel_name}: Failed to run")
        return False
    
    if ref_output.shape != kernel_output.shape:
        print(f"  [FAIL] {kernel_name}: Shape mismatch (ref: {ref_output.shape}, kernel: {kernel_output.shape})")
        return False
    
    # Check for exact match (quantization should be deterministic)
    matches = np.array_equal(ref_output, kernel_output)
    
    if matches:
        print(f"  [PASS] {kernel_name}: Output matches PyTorch exactly")
        return True
    else:
        # Count differences
        diff_mask = ref_output != kernel_output
        num_diffs = np.sum(diff_mask)
        abs_diffs = np.abs(ref_output.astype(np.int16) - kernel_output.astype(np.int16))
        max_diff = np.max(abs_diffs)
        
        # Check if differences are within tolerance
        # If scale is slightly different, ±1 differences are acceptable
        if max_diff <= tolerance:
            # Check if the differences are consistent with scale difference
            if ref_scale is not None and kernel_scale is not None:
                scale_diff_ratio = abs(ref_scale - kernel_scale) / ref_scale
                # If scale difference is small (< 0.1%) and max diff is ≤ 1, consider it acceptable
                if scale_diff_ratio < 0.001 and max_diff <= 1:
                    print(f"  [PASS] {kernel_name}: Output matches PyTorch (within tolerance)")
                    print(f"     {num_diffs}/{len(ref_output)} values differ by ±1 (due to small scale difference)")
                    return True
        
        print(f"  [FAIL] {kernel_name}: {num_diffs}/{len(ref_output)} values differ (max diff: {max_diff})")
        
        # Show first few differences
        if num_diffs > 0:
            diff_indices = np.where(diff_mask)[0][:5]
            print(f"     First differences:")
            for idx in diff_indices:
                print(f"       [{idx}]: ref={ref_output[idx]}, kernel={kernel_output[idx]}")
        
        return False

def verify_kernels(N=1024*1024, test_cases=None, seed=None):
    """Verify all quantization kernels against PyTorch.
    
    Args:
        N: Number of elements to test
        test_cases: Optional list of kernel names to test
        seed: Random seed (None for random, int for fixed seed, default: 42)
    """
    if seed is None:
        import random
        seed = random.randint(0, 2**31 - 1)
        print(f"Using random seed: {seed}")
    else:
        print(f"Using fixed seed: {seed}")
    
    print(f"Generating test data (N={N})...")
    x = generate_test_data(N, seed=seed)
    
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
                    print(f"  [WARN] Scale difference: {scale_diff:.6f} (ref: {ref_scale:.6f}, kernel: {kernel_scale:.6f})")
            
            is_correct = compare_outputs(ref_output, kernel_output, kernel_name, 
                                        ref_scale=ref_scale, kernel_scale=kernel_scale)
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
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}: {kernel_name}")
    
    if all_passed:
        print("\nAll kernels match PyTorch reference!")
    else:
        print("\nSome kernels have mismatches with PyTorch reference.")
    
    return all_passed

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Verify CUDA quantization kernels against PyTorch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default fixed seed (42)
  python3 verify_correctness.py
  
  # Use random seed each run
  python3 verify_correctness.py --random-seed
  
  # Use specific seed
  python3 verify_correctness.py --seed 123
  
  # Test specific kernels
  python3 verify_correctness.py quantize_naive_0 quantize_vectorized_1
        """
    )
    parser.add_argument('--N', type=int, default=1024*1024,
                       help='Number of elements (default: 1048576)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Fixed random seed (default: 42, use --random-seed for random)')
    parser.add_argument('--random-seed', action='store_true',
                       help='Use a random seed each run (overrides --seed)')
    parser.add_argument('test_cases', nargs='*',
                       help='Specific kernel names to test (e.g., quantize_naive_0)')
    
    args = parser.parse_args()
    
    seed = None if args.random_seed else args.seed
    
    success = verify_kernels(N=args.N, test_cases=args.test_cases if args.test_cases else None, seed=seed)
    sys.exit(0 if success else 1)
