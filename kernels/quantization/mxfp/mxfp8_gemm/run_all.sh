#!/bin/bash
#
# MXFP8 Tensor Core GEMM - Complete Profiling Suite
#
# This script:
# 1. Compiles the kernels
# 2. Runs benchmarks
# 3. Runs Nsight Compute profiling
# 4. Runs Nsight Systems tracing
# 5. Exports all results to the results/ directory
#
# Usage:
#   ./run_all.sh [--arch sm_80|sm_89|sm_90] [--size 4096] [--no-profile]
#

set -e

# Defaults
ARCH="sm_80"
SIZE=4096
WARMUP=10
ITERS=100
PROFILE=true
TRACE=true
OUTPUT_DIR="./results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --arch)
            ARCH="$2"
            shift 2
            ;;
        --size)
            SIZE="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --iters)
            ITERS="$2"
            shift 2
            ;;
        --no-profile)
            PROFILE=false
            shift
            ;;
        --no-trace)
            TRACE=false
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Header
echo "========================================================================"
echo "MXFP8 Tensor Core GEMM - Complete Profiling Suite"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  Architecture: $ARCH"
echo "  Matrix Size:  $SIZE"
echo "  Warmup:       $WARMUP"
echo "  Iterations:   $ITERS"
echo "  Profile:      $PROFILE"
echo "  Trace:        $TRACE"
echo "  Output Dir:   $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for CUDA
echo "Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA toolkit."
    exit 1
fi
nvcc --version | head -n 4

# Check for Python and PyTorch
echo ""
echo "Checking Python and PyTorch..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"N/A\"}')"

# Compile kernel
echo ""
echo "========================================================================"
echo "Step 1: Compiling CUDA Kernels"
echo "========================================================================"

KERNEL_SOURCE="mxfp8_gemm_combined.cu"
KERNEL_LIB="libmxfp8_gemm.so"

if [ -f "$KERNEL_LIB" ]; then
    echo "Kernel library exists. Recompiling..."
    rm -f "$KERNEL_LIB"
fi

nvcc -O3 -arch="$ARCH" -Xcompiler -fPIC -shared -lineinfo -o "$KERNEL_LIB" "$KERNEL_SOURCE"
echo "✓ Compiled: $KERNEL_LIB"

# Run benchmark
echo ""
echo "========================================================================"
echo "Step 2: Running Benchmarks"
echo "========================================================================"

PROFILE_ARGS=""
if [ "$PROFILE" = true ]; then
    PROFILE_ARGS="$PROFILE_ARGS --profile"
fi
if [ "$TRACE" = true ]; then
    PROFILE_ARGS="$PROFILE_ARGS --trace"
fi

python3 profile_and_benchmark.py \
    --size "$SIZE" \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --arch "$ARCH" \
    --output-dir "$OUTPUT_DIR" \
    $PROFILE_ARGS

# Additional NCU detailed analysis (if profiling enabled)
if [ "$PROFILE" = true ] && command -v ncu &> /dev/null; then
    echo ""
    echo "========================================================================"
    echo "Step 3: Detailed Nsight Compute Analysis"
    echo "========================================================================"
    
    # Create kernel runner for profiling
    RUNNER="$OUTPUT_DIR/kernel_runner_standalone.py"
    cat > "$RUNNER" << 'RUNNER_EOF'
#!/usr/bin/env python3
import torch
import ctypes
import sys
import os

# Get parameters
M = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
K = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
kernel_type = sys.argv[4] if len(sys.argv) > 4 else "tc"

# Find library
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lib_path = os.path.join(script_dir, "libmxfp8_gemm.so")

lib = ctypes.CDLL(lib_path)
lib.init_luts()

for fn_name in ['mxfp8_tc_gemm', 'naive_mxfp8_gemm']:
    fn = getattr(lib, fn_name)
    fn.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]

torch.manual_seed(42)
A_quant = torch.randint(0, 256, (M, K), dtype=torch.uint8, device='cuda')
A_scales = torch.rand(M, K // 32, dtype=torch.float32, device='cuda')
B_quant = torch.randint(0, 256, (K, N), dtype=torch.uint8, device='cuda')
B_scales = torch.rand(K // 32, N, dtype=torch.float32, device='cuda')
C = torch.zeros(M, N, dtype=torch.float32, device='cuda')

fn = lib.mxfp8_tc_gemm if kernel_type == "tc" else lib.naive_mxfp8_gemm

# Warmup
for _ in range(3):
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
print(f"Executed {kernel_type} kernel: {M}x{N}x{K}")
RUNNER_EOF
    chmod +x "$RUNNER"
    
    # Pad size
    M_PAD=$(( (SIZE + 127) / 128 * 128 ))
    N_PAD=$M_PAD
    K_PAD=$(( (SIZE + 31) / 32 * 32 ))
    
    echo "Running detailed NCU analysis for ${M_PAD}x${N_PAD}x${K_PAD}..."
    
    # Memory throughput analysis
    echo "  Analyzing memory throughput..."
    ncu --metrics \
        dram__bytes_read.sum,\
dram__bytes_write.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum,\
lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum \
        --csv \
        python3 "$RUNNER" $M_PAD $N_PAD $K_PAD tc \
        > "$OUTPUT_DIR/ncu_memory_metrics.csv" 2>/dev/null || true
    
    # Compute throughput analysis  
    echo "  Analyzing compute throughput..."
    ncu --metrics \
        sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_tensor.sum,\
sm__inst_executed.sum,\
sm__cycles_elapsed.avg \
        --csv \
        python3 "$RUNNER" $M_PAD $N_PAD $K_PAD tc \
        > "$OUTPUT_DIR/ncu_compute_metrics.csv" 2>/dev/null || true
    
    # Occupancy analysis
    echo "  Analyzing occupancy..."
    ncu --metrics \
        launch__waves_per_multiprocessor,\
launch__occupancy_per_register_count,\
launch__occupancy_per_shared_mem_size,\
launch__block_size,\
launch__grid_size \
        --csv \
        python3 "$RUNNER" $M_PAD $N_PAD $K_PAD tc \
        > "$OUTPUT_DIR/ncu_occupancy_metrics.csv" 2>/dev/null || true
    
    echo "✓ NCU metrics saved to $OUTPUT_DIR/ncu_*.csv"
fi

# Summary
echo ""
echo "========================================================================"
echo "COMPLETE"
echo "========================================================================"
echo ""
echo "Output files:"
ls -la "$OUTPUT_DIR/"
echo ""
echo "Key results:"
echo "  - benchmark_results.json : All timing and correctness data"
echo "  - benchmark_results.txt  : Human-readable summary"
if [ "$PROFILE" = true ]; then
    echo "  - *.ncu-rep              : Nsight Compute profiles (open with ncu-ui)"
    echo "  - ncu_*.csv              : Extracted NCU metrics"
fi
if [ "$TRACE" = true ]; then
    echo "  - *.nsys-rep             : Nsight Systems traces (open with nsys-ui)"
fi
echo ""
echo "To view NCU profile: ncu-ui $OUTPUT_DIR/profile_ncu_*.ncu-rep"
echo "To view NSYS trace:  nsys-ui $OUTPUT_DIR/profile_nsys_*.nsys-rep"
