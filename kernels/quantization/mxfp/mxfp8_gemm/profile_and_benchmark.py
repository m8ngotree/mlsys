#!/usr/bin/env python3
"""
MXFP8 Tensor Core GEMM - Profiling & Benchmarking Suite

Outputs all results to files for analysis:
- benchmark_results.json: Timing and performance metrics
- benchmark_results.txt: Human-readable summary
- profile_*.ncu-rep: Nsight Compute profiles
- profile_*.nsys-rep: Nsight Systems traces

Usage:
    python profile_and_benchmark.py [options]

Options:
    --size SIZE         Matrix size (default: 4096)
    --arch ARCH         CUDA arch: sm_80, sm_89, sm_90 (default: sm_80)
    --profile           Run Nsight Compute profiling
    --trace             Run Nsight Systems tracing
    --output-dir DIR    Output directory (default: ./results)
    --warmup N          Warmup iterations (default: 10)
    --iters N           Benchmark iterations (default: 100)
"""

import torch
import numpy as np
import ctypes
import subprocess
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, Optional


# ============================================================================
# MXFP8 E4M3 Quantization
# ============================================================================

class MXFP8Quantizer:
    """MXFP8 E4M3 quantization with per-block scaling."""
    
    BLOCK_SIZE = 32
    E4M3_MAX = 448.0
    
    @staticmethod
    def _build_e4m3_lut() -> torch.Tensor:
        lut = torch.zeros(256, dtype=torch.float32)
        for i in range(256):
            sign = (i >> 7) & 1
            exp = (i >> 3) & 0xF
            mant = i & 0x7
            
            if exp == 0:
                value = (mant / 8.0) * (2**-6)
            elif exp == 15 and mant != 0:
                value = float('nan')
            elif exp == 15 and mant == 0:
                value = 448.0
            else:
                value = (1.0 + mant / 8.0) * (2**(exp - 7))
            
            lut[i] = -value if sign else value
        return lut
    
    @classmethod
    def quantize(cls, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.dim() == 2
        M, K = x.shape
        assert K % cls.BLOCK_SIZE == 0
        
        x = x.float()
        num_blocks = K // cls.BLOCK_SIZE
        x_blocks = x.view(M, num_blocks, cls.BLOCK_SIZE)
        
        block_absmax = x_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scales = (block_absmax / cls.E4M3_MAX).squeeze(-1)
        
        x_scaled = x_blocks / block_absmax * cls.E4M3_MAX
        quantized = cls._float_to_e4m3(x_scaled.view(M, K))
        
        return quantized, scales
    
    @classmethod
    def _float_to_e4m3(cls, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        result = torch.zeros_like(x, dtype=torch.uint8)
        
        sign = (x < 0).to(torch.uint8)
        x_abs = x.abs().clamp(max=cls.E4M3_MAX)
        
        zero_mask = x_abs == 0
        subnormal_threshold = 2**-6
        subnormal_mask = (x_abs > 0) & (x_abs < subnormal_threshold)
        normal_mask = x_abs >= subnormal_threshold
        
        subnormal_mant = torch.round(x_abs / subnormal_threshold * 8).clamp(0, 7).to(torch.uint8)
        
        log2_x = torch.floor(torch.log2(x_abs.clamp(min=1e-45)))
        exp_unbiased = log2_x.clamp(-6, 8).to(torch.int32)
        exp_biased = (exp_unbiased + 7).clamp(0, 15).to(torch.uint8)
        significand = x_abs / (2.0 ** exp_unbiased) - 1.0
        mant = torch.round(significand * 8).clamp(0, 7).to(torch.uint8)
        
        normal_result = (sign << 7) | (exp_biased << 3) | mant
        subnormal_result = (sign << 7) | subnormal_mant
        zero_result = sign << 7
        
        result = torch.where(normal_mask, normal_result, result)
        result = torch.where(subnormal_mask, subnormal_result, result)
        result = torch.where(zero_mask, zero_result, result)
        
        return result
    
    @classmethod
    def dequantize(cls, quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        M, K = quantized.shape
        num_blocks = scales.shape[1]
        
        lut = cls._build_e4m3_lut().to(quantized.device)
        values = lut[quantized.long()]
        
        values = values.view(M, num_blocks, cls.BLOCK_SIZE)
        scales_expanded = scales.unsqueeze(-1)
        
        result = (values * scales_expanded).view(M, K)
        return result


# ============================================================================
# CUDA Kernel Wrapper
# ============================================================================

class MXFP8GEMMKernel:
    def __init__(self, lib_path: str):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Library not found: {lib_path}")
        
        self.lib = ctypes.CDLL(lib_path)
        self.lib.init_luts()
        
        for fn_name in ['mxfp8_tc_gemm', 'naive_mxfp8_gemm']:
            fn = getattr(self.lib, fn_name)
            fn.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
    
    def _call_kernel(self, fn_name: str, A_data, A_scales, B_data, B_scales, C):
        M, K = A_data.shape
        K2, N = B_data.shape
        assert K == K2
        
        fn = getattr(self.lib, fn_name)
        fn(
            ctypes.c_void_p(A_data.data_ptr()),
            ctypes.c_void_p(A_scales.data_ptr()),
            ctypes.c_void_p(B_data.data_ptr()),
            ctypes.c_void_p(B_scales.data_ptr()),
            ctypes.c_void_p(C.data_ptr()),
            M, N, K
        )
        torch.cuda.synchronize()
    
    def tensor_core_gemm(self, A_data, A_scales, B_data, B_scales, C):
        self._call_kernel('mxfp8_tc_gemm', A_data, A_scales, B_data, B_scales, C)
    
    def naive_gemm(self, A_data, A_scales, B_data, B_scales, C):
        self._call_kernel('naive_mxfp8_gemm', A_data, A_scales, B_data, B_scales, C)


# ============================================================================
# Compilation
# ============================================================================

def compile_kernel(source_path: str, output_path: str, arch: str = "sm_80") -> bool:
    cmd = ["nvcc", "-O3", f"-arch={arch}", "-Xcompiler", "-fPIC", 
           "-shared", "-lineinfo", "-o", output_path, source_path]
    
    print(f"Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}")
        return False
    
    print("✓ Compilation successful!")
    return True


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_fn(fn, warmup: int = 10, iters: int = 100) -> Tuple[float, float, float]:
    """Returns (mean_ms, min_ms, max_ms)"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        
        times.append(start.elapsed_time(end))
    
    return np.mean(times), np.min(times), np.max(times)


def compute_error_metrics(result: torch.Tensor, reference: torch.Tensor) -> Dict[str, float]:
    abs_err = (result - reference).abs()
    rel_err = abs_err / (reference.abs() + 1e-6)
    
    return {
        'max_abs_error': abs_err.max().item(),
        'mean_abs_error': abs_err.mean().item(),
        'max_rel_error': rel_err.max().item(),
        'mean_rel_error': rel_err.mean().item(),
        'rmse': torch.sqrt((abs_err ** 2).mean()).item(),
    }


# ============================================================================
# Profiling with Nsight Tools
# ============================================================================

def create_standalone_kernel_runner(script_dir: Path, lib_path: Path) -> Path:
    """Create a standalone script for profiling."""
    runner_code = f'''#!/usr/bin/env python3
"""Standalone kernel runner for Nsight profiling."""
import torch
import ctypes
import sys

M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
kernel_type = sys.argv[4]  # "tc" or "naive"

# Load kernel
lib = ctypes.CDLL("{lib_path}")
lib.init_luts()

for fn_name in ['mxfp8_tc_gemm', 'naive_mxfp8_gemm']:
    fn = getattr(lib, fn_name)
    fn.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]

# Create test data
torch.manual_seed(42)
A_quant = torch.randint(0, 256, (M, K), dtype=torch.uint8, device='cuda')
A_scales = torch.rand(M, K // 32, dtype=torch.float32, device='cuda')
B_quant = torch.randint(0, 256, (K, N), dtype=torch.uint8, device='cuda')
B_scales = torch.rand(K // 32, N, dtype=torch.float32, device='cuda')
C = torch.zeros(M, N, dtype=torch.float32, device='cuda')

# Warmup
fn = lib.mxfp8_tc_gemm if kernel_type == "tc" else lib.naive_mxfp8_gemm
for _ in range(5):
    fn(
        ctypes.c_void_p(A_quant.data_ptr()),
        ctypes.c_void_p(A_scales.data_ptr()),
        ctypes.c_void_p(B_quant.data_ptr()),
        ctypes.c_void_p(B_scales.data_ptr()),
        ctypes.c_void_p(C.data_ptr()),
        M, N, K
    )
    torch.cuda.synchronize()

# Profile run
fn(
    ctypes.c_void_p(A_quant.data_ptr()),
    ctypes.c_void_p(A_scales.data_ptr()),
    ctypes.c_void_p(B_quant.data_ptr()),
    ctypes.c_void_p(B_scales.data_ptr()),
    ctypes.c_void_p(C.data_ptr()),
    M, N, K
)
torch.cuda.synchronize()

print(f"Executed {{kernel_type}} kernel: {{M}}x{{N}}x{{K}}")
'''
    
    runner_path = script_dir / "kernel_runner.py"
    with open(runner_path, 'w') as f:
        f.write(runner_code)
    
    return runner_path


def run_ncu_profile(runner_path: Path, output_dir: Path, M: int, N: int, K: int, 
                    kernel_type: str = "tc") -> Optional[Path]:
    """Run Nsight Compute profiling."""
    output_file = output_dir / f"profile_ncu_{kernel_type}_{M}x{N}x{K}.ncu-rep"
    
    cmd = [
        "ncu",
        "--set", "full",
        "--target-processes", "all",
        "-o", str(output_file.with_suffix('')),  # ncu adds .ncu-rep
        "python3", str(runner_path), str(M), str(N), str(K), kernel_type
    ]
    
    print(f"\nRunning Nsight Compute: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"NCU failed: {result.stderr}")
        return None
    
    print(f"✓ NCU profile saved to: {output_file}")
    return output_file


def run_nsys_trace(runner_path: Path, output_dir: Path, M: int, N: int, K: int,
                   kernel_type: str = "tc") -> Optional[Path]:
    """Run Nsight Systems tracing."""
    output_file = output_dir / f"profile_nsys_{kernel_type}_{M}x{N}x{K}"
    
    cmd = [
        "nsys", "profile",
        "--stats=true",
        "--force-overwrite=true",
        "-o", str(output_file),
        "python3", str(runner_path), str(M), str(N), str(K), kernel_type
    ]
    
    print(f"\nRunning Nsight Systems: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"NSYS failed: {result.stderr}")
        return None
    
    print(f"✓ NSYS trace saved to: {output_file}.nsys-rep")
    return Path(f"{output_file}.nsys-rep")


def run_ncu_metrics(runner_path: Path, output_dir: Path, M: int, N: int, K: int,
                    kernel_type: str = "tc") -> Optional[Dict[str, Any]]:
    """Run Nsight Compute and extract key metrics."""
    
    cmd = [
        "ncu",
        "--metrics",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,"
        "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,"
        "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,"
        "sm__warps_active.avg.pct_of_peak_sustained_active,"
        "sm__inst_executed_pipe_tensor.sum,"
        "dram__bytes_read.sum,"
        "dram__bytes_write.sum",
        "--csv",
        "python3", str(runner_path), str(M), str(N), str(K), kernel_type
    ]
    
    print(f"\nExtracting NCU metrics...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"NCU metrics failed: {result.stderr}")
        return None
    
    # Parse CSV output
    metrics = {}
    lines = result.stdout.strip().split('\n')
    for line in lines:
        if ',' in line and not line.startswith('"ID"'):
            parts = line.split(',')
            if len(parts) >= 3:
                metric_name = parts[0].strip('"')
                metric_value = parts[-1].strip('"')
                try:
                    metrics[metric_name] = float(metric_value)
                except ValueError:
                    metrics[metric_name] = metric_value
    
    return metrics


# ============================================================================
# Main Benchmark
# ============================================================================

def run_full_benchmark(M: int, N: int, K: int, kernel: MXFP8GEMMKernel,
                       warmup: int, iters: int) -> Dict[str, Any]:
    """Run complete benchmark and return all results."""
    
    results = {
        'dimensions': {'M': M, 'N': N, 'K': K},
        'flops': 2 * M * N * K,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Generate test data
    torch.manual_seed(42)
    A_fp32 = torch.randn(M, K, device='cuda')
    B_fp32 = torch.randn(K, N, device='cuda')
    
    # Quantize
    A_quant, A_scales = MXFP8Quantizer.quantize(A_fp32)
    B_quant, B_scales = MXFP8Quantizer.quantize(B_fp32)
    
    A_quant = A_quant.cuda().contiguous()
    A_scales = A_scales.cuda().contiguous()
    B_quant = B_quant.cuda().contiguous()
    B_scales = B_scales.cuda().contiguous()
    
    C_tc = torch.zeros(M, N, device='cuda', dtype=torch.float32)
    C_naive = torch.zeros(M, N, device='cuda', dtype=torch.float32)
    
    A_dequant = MXFP8Quantizer.dequantize(A_quant.cpu(), A_scales.cpu()).cuda()
    B_dequant = MXFP8Quantizer.dequantize(B_quant.cpu(), B_scales.cpu()).cuda()
    
    # FP16 cuBLAS
    A_fp16, B_fp16 = A_fp32.half(), B_fp32.half()
    mean, min_t, max_t = benchmark_fn(lambda: torch.mm(A_fp16, B_fp16), warmup, iters)
    C_fp16_ref = torch.mm(A_fp16, B_fp16).float()
    
    results['fp16_cublas'] = {
        'mean_ms': mean, 'min_ms': min_t, 'max_ms': max_t,
        'tflops': results['flops'] / (mean * 1e9),
    }
    
    # PyTorch MXFP8
    def pytorch_mxfp8():
        A_deq = MXFP8Quantizer.dequantize(A_quant.cpu(), A_scales.cpu()).cuda()
        B_deq = MXFP8Quantizer.dequantize(B_quant.cpu(), B_scales.cpu()).cuda()
        return torch.mm(A_deq, B_deq)
    
    mean, min_t, max_t = benchmark_fn(pytorch_mxfp8, warmup, iters)
    C_pytorch = pytorch_mxfp8()
    
    results['pytorch_mxfp8'] = {
        'mean_ms': mean, 'min_ms': min_t, 'max_ms': max_t,
        'tflops': results['flops'] / (mean * 1e9),
    }
    
    # Naive CUDA
    mean, min_t, max_t = benchmark_fn(
        lambda: kernel.naive_gemm(A_quant, A_scales, B_quant, B_scales, C_naive),
        warmup, iters
    )
    kernel.naive_gemm(A_quant, A_scales, B_quant, B_scales, C_naive)
    
    results['naive_cuda'] = {
        'mean_ms': mean, 'min_ms': min_t, 'max_ms': max_t,
        'tflops': results['flops'] / (mean * 1e9),
    }
    
    # Tensor Core
    mean, min_t, max_t = benchmark_fn(
        lambda: kernel.tensor_core_gemm(A_quant, A_scales, B_quant, B_scales, C_tc),
        warmup, iters
    )
    kernel.tensor_core_gemm(A_quant, A_scales, B_quant, B_scales, C_tc)
    
    results['tensor_core'] = {
        'mean_ms': mean, 'min_ms': min_t, 'max_ms': max_t,
        'tflops': results['flops'] / (mean * 1e9),
    }
    
    # Speedups
    results['speedups'] = {
        'tc_vs_fp16': results['fp16_cublas']['mean_ms'] / results['tensor_core']['mean_ms'],
        'tc_vs_pytorch': results['pytorch_mxfp8']['mean_ms'] / results['tensor_core']['mean_ms'],
        'tc_vs_naive': results['naive_cuda']['mean_ms'] / results['tensor_core']['mean_ms'],
    }
    
    # Correctness
    C_ref = torch.mm(A_dequant, B_dequant)
    
    results['correctness'] = {
        'naive_vs_ref': compute_error_metrics(C_naive, C_ref),
        'tc_vs_ref': compute_error_metrics(C_tc, C_ref),
        'tc_vs_naive': compute_error_metrics(C_tc, C_naive),
        'tc_vs_fp16': compute_error_metrics(C_tc, C_fp16_ref),
    }
    
    return results


def write_results_txt(results: Dict[str, Any], output_path: Path):
    """Write human-readable results."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MXFP8 TENSOR CORE GEMM - BENCHMARK RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {results.get('timestamp', 'N/A')}\n")
        f.write(f"Device: {results.get('device', 'N/A')}\n")
        f.write(f"CUDA Version: {results.get('cuda_version', 'N/A')}\n\n")
        
        for size_result in results.get('benchmarks', []):
            dims = size_result['dimensions']
            f.write("-" * 80 + "\n")
            f.write(f"Matrix Size: {dims['M']} x {dims['N']} x {dims['K']}\n")
            f.write(f"Total FLOPs: {size_result['flops']:,}\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("PERFORMANCE:\n")
            f.write(f"{'Kernel':<25} {'Mean (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'TFLOPS':<12}\n")
            f.write("-" * 75 + "\n")
            
            for kernel_name in ['fp16_cublas', 'pytorch_mxfp8', 'naive_cuda', 'tensor_core']:
                data = size_result.get(kernel_name, {})
                f.write(f"{kernel_name:<25} "
                       f"{data.get('mean_ms', 0):<12.4f} "
                       f"{data.get('min_ms', 0):<12.4f} "
                       f"{data.get('max_ms', 0):<12.4f} "
                       f"{data.get('tflops', 0):<12.2f}\n")
            
            f.write("\nSPEEDUPS:\n")
            speedups = size_result.get('speedups', {})
            f.write(f"  Tensor Core vs FP16 cuBLAS: {speedups.get('tc_vs_fp16', 0):.2f}x\n")
            f.write(f"  Tensor Core vs PyTorch:     {speedups.get('tc_vs_pytorch', 0):.2f}x\n")
            f.write(f"  Tensor Core vs Naive CUDA:  {speedups.get('tc_vs_naive', 0):.2f}x\n")
            
            f.write("\nCORRECTNESS:\n")
            correctness = size_result.get('correctness', {})
            for name, metrics in correctness.items():
                f.write(f"  {name}:\n")
                f.write(f"    Max Abs Error:  {metrics.get('max_abs_error', 0):.6f}\n")
                f.write(f"    Mean Rel Error: {metrics.get('mean_rel_error', 0):.6f}\n")
                f.write(f"    RMSE:           {metrics.get('rmse', 0):.6f}\n")
            
            f.write("\n")
        
        # NCU metrics if available
        if 'ncu_metrics' in results:
            f.write("=" * 80 + "\n")
            f.write("NSIGHT COMPUTE METRICS\n")
            f.write("=" * 80 + "\n\n")
            
            for size_key, metrics in results['ncu_metrics'].items():
                f.write(f"{size_key}:\n")
                for metric_name, value in metrics.items():
                    f.write(f"  {metric_name}: {value}\n")
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(description='MXFP8 GEMM Profiling & Benchmarking')
    parser.add_argument('--size', type=int, default=4096, help='Square matrix size')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--arch', type=str, default='sm_80')
    parser.add_argument('--profile', action='store_true', help='Run Nsight Compute profiling')
    parser.add_argument('--trace', action='store_true', help='Run Nsight Systems tracing')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--compile-only', action='store_true')
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    source_path = script_dir / "mxfp8_gemm_combined.cu"
    lib_path = script_dir / "libmxfp8_gemm.so"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compile
    if not lib_path.exists():
        print(f"Compiling kernel for {args.arch}...")
        if not compile_kernel(str(source_path), str(lib_path), args.arch):
            return 1
    
    if args.compile_only:
        print("Compilation complete.")
        return 0
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return 1
    
    # Device info
    device_name = torch.cuda.get_device_name()
    cuda_version = torch.version.cuda
    
    print(f"\nDevice: {device_name}")
    print(f"CUDA Version: {cuda_version}")
    print(f"Output Directory: {output_dir}")
    
    # Load kernel
    kernel = MXFP8GEMMKernel(str(lib_path))
    
    # Test sizes
    sizes = [
        (args.size, args.size, args.size),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (4096, 4096, 11008),
        (4096, 11008, 4096),
    ]
    
    # Remove duplicates
    seen = set()
    unique_sizes = []
    for s in sizes:
        if s not in seen:
            seen.add(s)
            unique_sizes.append(s)
    
    # Results container
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'device': device_name,
        'cuda_version': cuda_version,
        'arch': args.arch,
        'warmup': args.warmup,
        'iters': args.iters,
        'benchmarks': [],
        'ncu_metrics': {},
    }
    
    # Run benchmarks
    for M, N, K in unique_sizes:
        M_pad = ((M + 127) // 128) * 128
        N_pad = ((N + 127) // 128) * 128
        K_pad = ((K + 31) // 32) * 32
        
        print(f"\n{'='*70}")
        print(f"Benchmarking: {M_pad} x {N_pad} x {K_pad}")
        print(f"{'='*70}")
        
        results = run_full_benchmark(M_pad, N_pad, K_pad, kernel, args.warmup, args.iters)
        all_results['benchmarks'].append(results)
        
        # Print summary
        tc = results['tensor_core']
        fp16 = results['fp16_cublas']
        print(f"\n  Tensor Core: {tc['mean_ms']:.4f} ms ({tc['tflops']:.2f} TFLOPS)")
        print(f"  FP16 cuBLAS: {fp16['mean_ms']:.4f} ms ({fp16['tflops']:.2f} TFLOPS)")
        print(f"  Speedup: {results['speedups']['tc_vs_fp16']:.2f}x")
    
    # Profiling
    if args.profile or args.trace:
        runner_path = create_standalone_kernel_runner(script_dir, lib_path)
        
        # Profile first size only (to save time)
        M, N, K = unique_sizes[0]
        M_pad = ((M + 127) // 128) * 128
        N_pad = ((N + 127) // 128) * 128
        K_pad = ((K + 31) // 32) * 32
        
        if args.profile:
            # Full NCU profile
            run_ncu_profile(runner_path, output_dir, M_pad, N_pad, K_pad, "tc")
            
            # Extract metrics
            metrics = run_ncu_metrics(runner_path, output_dir, M_pad, N_pad, K_pad, "tc")
            if metrics:
                all_results['ncu_metrics'][f"{M_pad}x{N_pad}x{K_pad}"] = metrics
        
        if args.trace:
            run_nsys_trace(runner_path, output_dir, M_pad, N_pad, K_pad, "tc")
    
    # Save results
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ JSON results saved to: {json_path}")
    
    txt_path = output_dir / "benchmark_results.txt"
    write_results_txt(all_results, txt_path)
    print(f"✓ Text results saved to: {txt_path}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Size':<25} {'TC (ms)':<12} {'FP16 (ms)':<12} {'Speedup':<10} {'TC TFLOPS':<12}")
    print("-" * 70)
    
    for result in all_results['benchmarks']:
        dims = result['dimensions']
        size_str = f"{dims['M']}x{dims['N']}x{dims['K']}"
        tc = result['tensor_core']
        fp16 = result['fp16_cublas']
        speedup = result['speedups']['tc_vs_fp16']
        print(f"{size_str:<25} {tc['mean_ms']:<12.4f} {fp16['mean_ms']:<12.4f} {speedup:<10.2f}x {tc['tflops']:<12.2f}")
    
    print(f"\nAll results saved to: {output_dir}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
