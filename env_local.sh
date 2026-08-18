#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
source "$ROOT/.venv/bin/activate"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$ROOT/outputs/train}"
# Point CUDA_HOME at pip nvidia cuda_nvrtc so TransformerEngine can find libnvrtc
export CUDA_HOME="$ROOT/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
export CUDA_PATH="$CUDA_HOME"
NVIDIA_LIBS=$(find "$ROOT/.venv/lib/python3.10/site-packages/nvidia" -type d -name lib 2>/dev/null | paste -sd:)
export LD_LIBRARY_PATH="$ROOT/.venv/lib/python3.10/site-packages/torch/lib:${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"
# Ensure unversioned libcudart.so exists for ctypes.CDLL("libcudart.so")
CUDA_RT_LIB="$ROOT/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
if [[ -e "$CUDA_RT_LIB/libcudart.so.12" && ! -e "$CUDA_RT_LIB/libcudart.so" ]]; then
  ln -sf libcudart.so.12 "$CUDA_RT_LIB/libcudart.so"
fi
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=8
