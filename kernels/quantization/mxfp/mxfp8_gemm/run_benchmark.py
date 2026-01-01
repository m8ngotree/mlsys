#!/usr/bin/env python3
"""
MXFP8 Tensor Core GEMM - Complete Benchmark Suite

Usage:
    python run_benchmark.py [--size SIZE] [--warmup WARMUP] [--iters ITERS] [--arch ARCH]

This script:
1. Compiles the CUDA kernels
2. Runs correctness tests
3. Benchmarks against cuBLAS FP16 and PyTorch
4. Reports performance metrics and speedups
"""

import torch
import numpy as np
import ctypes
import subprocess
import argparse
from pathlib import Path
from typing import Tuple
import os


# ============================================================================
# MXFP8 E4M3 Quantization
# ============================================================================

class MXFP8Quantizer:
    """MXFP8 E4M3 quantization with per-block scaling."""
    
    BLOCK_SIZE = 32
    E4M3_MAX = 448.0
    
    @staticmethod
    def _build_e4m3_lut() -> torch.Tensor:
        """Build lookup table for E4M3 -> FP32 conversion."""
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
        """Quantize FP32/FP16 tensor to MXFP8 E4M3 format."""
        assert x.dim() == 2, "Expected 2D tensor"
        M, K = x.shape
        assert K % cls.BLOCK_SIZE == 0, f"K ({K}) must be divisible by {cls.BLOCK_SIZE}"
        
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
        """Convert FP32 to E4M3 uint8."""
        x = x.float()
        result = torch.zeros_like(x, dtype=torch.uint8)
        
        sign = (x < 0).to(torch.uint8)
        x_abs = x.abs().clamp(max=cls.E4M3_MAX)
        
        zero_mask = x_abs == 0
        subnormal_threshold = 2**-6
        subnormal_mask = (x_abs > 0) & (x_abs < subnormal_threshold)
        normal_mask = x_abs >= subnormal_threshold
        
        # Subnormals
        subnormal_mant = torch.round(x_abs / subnormal_threshold * 8).clamp(0, 7).to(torch.uint8)
        
        # Normals
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
        """Dequantize MXFP8 back to FP32."""
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
    """CUDA kernel wrapper."""
    
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


def compile_kernel(source_path: str, output_path: str, arch: str = "sm_80") -> bool:
    """Compile CUDA kernel."""
    cmd = ["nvcc", "-O3", f"-arch={arch}", "-Xcompiler", "-fPIC", "-shared", "-o", output_path, source_path]
    
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

def benchmark_fn(fn, warmup: int = 10, iters: int = 100) -> float:
    """Benchmark a function, return average time in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / iters


def compute_error_metrics(result: torch.Tensor, reference: torch.Tensor) -> dict:
    abs_err = (result - reference).abs()
    rel_err = abs_err / (reference.abs() + 1e-6)
    
    return {
        'max_abs': abs_err.max().item(),
        'mean_abs': abs_err.mean().item(),
        'max_rel': rel_err.max().item(),
        'mean_rel': rel_err.mean().item(),
    }


def run_benchmark(M: int, N: int, K: int, kernel: MXFP8GEMMKernel,
                  warmup: int = 10, iters: int = 100) -> dict:
    """Run full benchmark suite."""
    
    print(f"\n{'='*70}")
    print(f"MXFP8 Tensor Core GEMM Benchmark: M={M}, N={N}, K={K}")
    print(f"{'='*70}")
    
    torch.manual_seed(42)
    A_fp32 = torch.randn(M, K, device='cuda')
    B_fp32 = torch.randn(K, N, device='cuda')
    
    print("\n[1] Quantizing matrices to MXFP8 E4M3...")
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
    
    results = {}
    
    # FP16 cuBLAS
    print("\n[2] Benchmarking FP16 cuBLAS (baseline)...")
    A_fp16, B_fp16 = A_fp32.half(), B_fp32.half()
    fp16_time = benchmark_fn(lambda: torch.mm(A_fp16, B_fp16), warmup, iters)
    C_fp16_ref = torch.mm(A_fp16, B_fp16).float()
    print(f"    Time: {fp16_time:.4f} ms")
    results['fp16_cublas'] = fp16_time
    
    # PyTorch MXFP8
    print("\n[3] Benchmarking PyTorch MXFP8 (dequant → matmul)...")
    
    def pytorch_mxfp8():
        A_deq = MXFP8Quantizer.dequantize(A_quant.cpu(), A_scales.cpu()).cuda()
        B_deq = MXFP8Quantizer.dequantize(B_quant.cpu(), B_scales.cpu()).cuda()
        return torch.mm(A_deq, B_deq)
    
    pytorch_time = benchmark_fn(pytorch_mxfp8, warmup, iters)
    C_pytorch = pytorch_mxfp8()
    print(f"    Time: {pytorch_time:.4f} ms")
    results['pytorch_mxfp8'] = pytorch_time
    
    # Naive CUDA
    print("\n[4] Benchmarking Naive CUDA MXFP8...")
    naive_time = benchmark_fn(
        lambda: kernel.naive_gemm(A_quant, A_scales, B_quant, B_scales, C_naive),
        warmup, iters
    )
    kernel.naive_gemm(A_quant, A_scales, B_quant, B_scales, C_naive)
    print(f"    Time: {naive_time:.4f} ms")
    results['naive_cuda'] = naive_time
    
    # Tensor Core
    print("\n[5] Benchmarking Tensor Core MXFP8...")
    tc_time = benchmark_fn(
        lambda: kernel.tensor_core_gemm(A_quant, A_scales, B_quant, B_scales, C_tc),
        warmup, iters
    )
    kernel.tensor_core_gemm(A_quant, A_scales, B_quant, B_scales, C_tc)
    print(f"    Time: {tc_time:.4f} ms")
    results['tensor_core'] = tc_time
    
    # Performance Summary
    flops = 2.0 * M * N * K
    
    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Kernel':<25} {'Time (ms)':<12} {'TFLOPS':<12} {'vs FP16':<12}")
    print(f"{'-'*70}")
    
    for name, time_ms in [
        ('FP16 cuBLAS', fp16_time),
        ('PyTorch MXFP8', pytorch_time),
        ('Naive CUDA', naive_time),
        ('Tensor Core MXFP8', tc_time),
    ]:
        tflops = flops / (time_ms * 1e9)
        vs_fp16 = fp16_time / time_ms
        print(f"{name:<25} {time_ms:<12.4f} {tflops:<12.2f} {vs_fp16:<12.2f}x")
    
    # Correctness
    print(f"\n{'='*70}")
    print("CORRECTNESS VERIFICATION")
    print(f"{'='*70}")
    
    C_ref = torch.mm(A_dequant, B_dequant)
    
    print("\n→ Comparing against dequantized reference:")
    for name, result in [('Naive CUDA', C_naive), ('Tensor Core', C_tc)]:
        metrics = compute_error_metrics(result, C_ref)
        status = "✓ PASS" if metrics['max_abs'] < 1e-2 else "✗ FAIL"
        print(f"  {name}: max_abs={metrics['max_abs']:.6f}, mean_rel={metrics['mean_rel']:.6f} {status}")
    
    print("\n→ Quantization error vs FP16:")
    for name, result in [('Tensor Core', C_tc)]:
        metrics = compute_error_metrics(result, C_fp16_ref)
        print(f"  {name}: max_rel={metrics['max_rel']:.4f}, mean_rel={metrics['mean_rel']:.4f}")
    
    print("\n→ TC vs Naive consistency:")
    tc_vs_naive = compute_error_metrics(C_tc, C_naive)
    status = "✓ MATCH" if tc_vs_naive['max_abs'] < 1e-3 else "✗ MISMATCH"
    print(f"  max_abs={tc_vs_naive['max_abs']:.6f} {status}")
    
    results['speedup_vs_fp16'] = fp16_time / tc_time
    results['speedup_vs_pytorch'] = pytorch_time / tc_time
    
    return results


def main():
    parser = argparse.ArgumentParser(description='MXFP8 Tensor Core GEMM Benchmark')
    parser.add_argument('--size', type=int, default=4096, help='Square matrix size')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--arch', type=str, default='sm_80', help='sm_80, sm_89, sm_90')
    parser.add_argument('--compile-only', action='store_true')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    source_path = script_dir / "mxfp8_gemm_combined.cu"
    lib_path = script_dir / "libmxfp8_gemm.so"
    
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
    
    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    kernel = MXFP8GEMMKernel(str(lib_path))
    
    sizes = [
        (args.size, args.size, args.size),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (4096, 4096, 11008),
        (4096, 11008, 4096),
    ]
    
    seen = set()
    unique_sizes = []
    for s in sizes:
        if s not in seen:
            seen.add(s)
            unique_sizes.append(s)
    
    all_results = []
    for M, N, K in unique_sizes:
        M_pad = ((M + 127) // 128) * 128
        N_pad = ((N + 127) // 128) * 128
        K_pad = ((K + 31) // 32) * 32
        
        results = run_benchmark(M_pad, N_pad, K_pad, kernel, args.warmup, args.iters)
        all_results.append((M_pad, N_pad, K_pad, results))
    
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Size':<25} {'TC Time (ms)':<15} {'FP16 Time (ms)':<15} {'Speedup':<10}")
    print(f"{'-'*70}")
    
    for M, N, K, results in all_results:
        size_str = f"{M}x{N}x{K}"
        tc_time = results['tensor_core']
        fp16_time = results['fp16_cublas']
        speedup = results['speedup_vs_fp16']
        print(f"{size_str:<25} {tc_time:<15.4f} {fp16_time:<15.4f} {speedup:<10.2f}x")
    
    return 0


if __name__ == "__main__":
    exit(main())
