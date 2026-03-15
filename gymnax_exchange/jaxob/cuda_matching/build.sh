#!/bin/bash
# Build CUDA matching kernel for JAX FFI
#
# Must run on a compute node with nvcc available:
#   srun --nodes=1 --gpus-per-node=1 --time=00:10:00 --account=brics.s5e bash build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure conda + python are on PATH
export CONDA_PREFIX=${CONDA_PREFIX:-/projects/s5e/quant/miniforge3}
export PATH=$CONDA_PREFIX/bin:$PATH

# Load CUDA module if available (compute nodes)
module load cuda/12.6 2>/dev/null || true

# Get XLA FFI include path from JAX
XLA_INCLUDE=$(python3 -c "import jax; print(jax.ffi.include_dir())")
echo "XLA FFI include: ${XLA_INCLUDE}"

# Detect GPU architecture
GPU_ARCH=${GPU_ARCH:-sm_90}
echo "GPU architecture: ${GPU_ARCH}"

echo "Compiling cuda_matching_kernel.cu ..."
nvcc -shared -o libcuda_matching.so \
    -std=c++17 \
    -Xcompiler -fPIC \
    -I"${XLA_INCLUDE}" \
    --gpu-architecture="${GPU_ARCH}" \
    -O3 \
    cuda_matching_kernel.cu

echo "Build successful:"
ls -lh "${SCRIPT_DIR}/libcuda_matching.so"
