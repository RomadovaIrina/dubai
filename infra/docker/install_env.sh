#!/bin/bash
# DabAI image (r15 T): the software of the ML worker, EXACTLY the validated r14-d environments (pinned locks, no resolver), in steps.
# Used by infra/docker/Dockerfile (VENV_ROOT=/venv, DUB_ROOT=/workspace/dub/dubai) and by validate_recipe.sh (a clean room on a GPU server).
#     bash infra/docker/install_env.sh <step> ...      steps: venvs | latentsync | llama_cuda | api | check
#   venvs       /venv/dabai (pipeline, requirements.dabai.lock.txt: torch 2.7.1+cu128, Chatterbox @5de7a54, ...), /venv/mediapipe, /venv/main (NVML)
#   latentsync  third_party/latentsync at the pinned commit (code Apache-2.0; its checkpoints/ are symlinks into models/, made at start)
#   llama_cuda  llama-cpp-python 0.3.35 built for CUDA sm_120 into third_party/llama_cuda (needs nvcc >= 12.8; ~10 min with 16 jobs)
#   api         /venv/api: FastAPI + uvicorn for scripts/job/job_api.py (kept out of the ML venv)
#   check       imports + versions of every venv (no GPU needed)
# Never installed: CodeFormer (S-Lab NC: code + weights), htdemucs weights, InsightFace / RetinaFace weights (NC). See INTERFACE.md "Image".
set -euo pipefail
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; DUB_ROOT=${DUB_ROOT:-$(cd "$H/../.." && pwd)}; V=${VENV_ROOT:-/venv}; PYV=${PYTHON_VERSION:-3.11.17}
export UV_NO_CACHE=${UV_NO_CACHE:-1} UV_LINK_MODE=copy UV_HTTP_TIMEOUT=600 UV_PYTHON_PREFERENCE=only-managed
log(){ echo "[$(date +%T)] == $*"; }
TORCH_IDX=https://download.pytorch.org/whl/cu128; LLAMA_CPU_IDX=https://abetlen.github.io/llama-cpp-python/whl/cpu

venvs(){
  local L=$H/requirements.dabai.lock.txt P=$V/dabai/bin/python
  log "/venv/dabai from $(basename $L) ($(grep -c . $L) pins)"
  [ -x $P ] || uv venv -q --python $PYV --seed $V/dabai
  # pass 1: every pin except the sdist that needs its build deps installed first (insightface 0.7.3: cython + numpy, no build isolation)
  grep -v '^insightface==' $L > /tmp/dabai.pass1.txt
  uv pip install --python $P --no-deps --index-strategy unsafe-best-match --extra-index-url $TORCH_IDX --extra-index-url $LLAMA_CPU_IDX -r /tmp/dabai.pass1.txt
  uv pip install --python $P --no-deps --no-build-isolation "$(grep '^insightface==' $L)"   # code MIT; its NC weights are never fetched (tripwires)
  log "/venv/mediapipe"; [ -x $V/mediapipe/bin/python ] || uv venv -q --python $PYV --seed $V/mediapipe
  uv pip install --python $V/mediapipe/bin/python --no-deps -r $H/requirements.mediapipe.lock.txt
  log "/venv/main (NVML sampler)"; [ -x $V/main/bin/python ] || uv venv -q --python $PYV --seed $V/main
  uv pip install --python $V/main/bin/python --no-deps -r $H/requirements.main.lock.txt
}
latentsync(){
  local d=$DUB_ROOT/third_party/latentsync c=${LATENTSYNC_COMMIT:-a229c3948406bc2cf6eaf4873e662e70c6a04746}
  log "LatentSync code @${c:0:10}"; mkdir -p $d; cd $d
  [ -d .git ] || { git init -q .; git remote add origin https://github.com/bytedance/LatentSync; }
  git fetch -q --depth 1 origin $c && git checkout -q FETCH_HEAD && mkdir -p checkpoints && git rev-parse HEAD
}
llama_cuda(){
  local O=$DUB_ROOT/third_party/llama_cuda; mkdir -p $O; log "llama-cpp-python 0.3.35 CUDA sm_120 -> $O ($(${CUDACXX:-/usr/local/cuda/bin/nvcc} --version | tail -1))"
  CUDACXX=${CUDACXX:-/usr/local/cuda/bin/nvcc} CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120" CMAKE_BUILD_PARALLEL_LEVEL=${JOBS:-16} FORCE_CMAKE=1 \
    uv pip install --python $V/dabai/bin/python --target $O --no-deps --reinstall llama-cpp-python==0.3.35
}
api(){
  log "/venv/api"; [ -x $V/api/bin/python ] || uv venv -q --python $PYV $V/api
  uv pip install --python $V/api/bin/python --no-deps -r $H/requirements.api.lock.txt
}
check(){
  $V/dabai/bin/python - <<'PY'
import importlib, importlib.metadata as m
for n in ("torch", "torchaudio", "torchvision", "diffusers", "transformers", "faster_whisper", "pyannote.audio", "chatterbox", "cv2", "av", "onnxruntime", "librosa", "silero_vad", "llama_cpp"):
    importlib.import_module(n)
import torch; print("dabai ok: torch", torch.__version__, "cuda build", torch.version.cuda, "| chatterbox", m.version("chatterbox-tts"), "| cv2", __import__("cv2").__version__)
PY
  $V/mediapipe/bin/python -c "import mediapipe as mp; print('mediapipe ok', mp.__version__)"
  $V/main/bin/python -c "import pynvml; print('nvml ok')"
  [ -x $V/api/bin/python ] && $V/api/bin/python -c "import fastapi, uvicorn; print('api ok: fastapi', fastapi.__version__, 'uvicorn', uvicorn.__version__)" || true
  [ -d $DUB_ROOT/third_party/llama_cuda/llama_cpp ] && PYTHONPATH=$DUB_ROOT/third_party/llama_cuda $V/dabai/bin/python -c "import llama_cpp; print('llama_cuda', llama_cpp.__version__, 'gpu offload', llama_cpp.llama_supports_gpu_offload())" || true
}

for s in "$@"; do "$s"; done
