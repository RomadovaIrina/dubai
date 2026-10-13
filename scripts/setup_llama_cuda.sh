#!/bin/bash
# CUDA (sm_120) build of the SAME llama-cpp-python as the production env (40-llama.txt), installed into a separate directory so the CPU
# build in /venv/dabai stays untouched. Used by run_clean_pipeline_05.py --qwen-gpu-layers -1 with this directory first on PYTHONPATH:
#   PYTHONPATH=third_party/llama_cuda:$PYTHONPATH python scripts/pilot/run_clean_pipeline_05.py ... --qwen-gpu-layers -1
set -euo pipefail
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PY=/venv/dabai/bin/python
VER=$(grep -oP 'llama-cpp-python==\K[0-9.]+' "${DUB_ROOT}/scripts/requirements/40-llama.txt" || echo 0.3.35)
OUT="${DUB_ROOT}/third_party/llama_cuda"; mkdir -p "$OUT" "${DUB_ROOT}/logs"
export CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120" CMAKE_BUILD_PARALLEL_LEVEL="${JOBS:-12}" FORCE_CMAKE=1
uv pip install --python "$PY" --target "$OUT" --no-deps --no-cache --reinstall "llama-cpp-python==${VER}" 2>&1 | tail -5
PYTHONPATH="$OUT" "$PY" -c "import llama_cpp; print('llama_cpp', llama_cpp.__version__, llama_cpp.__file__, 'gpu_offload', llama_cpp.llama_supports_gpu_offload())"
