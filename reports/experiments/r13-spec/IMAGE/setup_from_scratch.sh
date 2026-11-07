#!/bin/bash
# r13 spec 5.1: the CURRENT clean DabAI pipeline from scratch on an empty CUDA 12.8 / Ubuntu 24.04 / RTX 5090 machine (the r13-t recipe, in order).
# It only installs and downloads; it builds no image and deploys nothing (infra does that). Idempotent: every step skips what exists.
#     HF_TOKEN=<token with the pyannote 3.1 terms accepted> bash reports/experiments/r13-spec/IMAGE/setup_from_scratch.sh
# Layout: repo /workspace/dub/dubai (models/ and third_party/ inside it), venvs /venv/dabai (pipeline), /venv/mediapipe, /venv/main (NVML sampler),
# NC quarantine /workspace/dub/quarantine_nc (htdemucs), /workspace/dub/sep_pylib (demucs code). For a network volume: everything under
# /workspace/dub + the two caches below must be on it (5.1: no download at start).
set -euo pipefail
R=${DUB_ROOT:-/workspace/dub/dubai}; cd "$R"; export DUB_ROOT=$R
: "${HF_TOKEN:?HF_TOKEN is needed (gated pyannote/speaker-diarization-3.1)}"
mkdir -p /workspace/dub; [ -f /workspace/dub/.env ] || { printf 'HF_TOKEN=%s\n' "$HF_TOKEN" > /workspace/dub/.env; chmod 600 /workspace/dub/.env; }
step(){ echo "[$(date +%T)] == $*"; }

step "1/8 /venv/dabai: torch 2.7.1+cu128, the pinned requirements, Chatterbox at the pinned commit (scripts/setup_dabai.sh, stages 0-9)"
CHATTERBOX_REF=${CHATTERBOX_REF:-5de7a54aa4e5e2baadb0182dde554908b48b85c2} bash scripts/setup_dabai.sh
step "2/8 models: whisper large-v3, Chatterbox v3, LatentSync 1.6, sd-vae, wav2vec2, MediaPipe task (scripts/download_models.sh; Qwen 7B skipped: unused)"
DUB_SKIP_QWEN7B=${DUB_SKIP_QWEN7B:-1} bash scripts/download_models.sh
step "3/8 /venv/mediapipe (scripts/setup_mediapipe.sh)"
bash scripts/setup_mediapipe.sh
step "4/8 llama-cpp-python 0.3.35 built for CUDA sm_120 into third_party/llama_cuda (scripts/setup_llama_cuda.sh, ~10 min with 16 jobs)"
JOBS=${JOBS:-16} bash scripts/setup_llama_cuda.sh
step "5/8 Qwen 14B Q5_K_M, CodeFormer (NC), Real-ESRGAN x2, demucs 4.0.1 + htdemucs (NC quarantine), Spleeter 2stems, noisereduce (scripts/r9/r9_models.sh)"
bash scripts/r9/r9_models.sh
step "6/8 runtime caches fetched NOW, not at the first job: pyannote 3.1 (-> ~/.cache/torch/pyannote) and the pkuseg model Chatterbox loads (-> ~/.pkuseg)"
. scripts/env.sh >/dev/null
/venv/dabai/bin/python - <<'PY'
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
from pyannote.audio import Pipeline
p = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.environ["HF_TOKEN"]); assert p is not None, "accept the pyannote terms for this token"
print("pyannote ok")
try:
    import pkuseg; pkuseg.pkuseg(); print("pkuseg ok")   # Chatterbox's Chinese segmenter: downloads spacy_ontonotes on first use
except Exception as e:
    print("pkuseg prefetch skipped:", e)
PY
step "7/8 /venv/main: nvidia-ml-py for the NVML GPU sampler (scripts/r10/gpu_sampler.py; the 1:5 GPU accounting)"
uv pip install --python /venv/main/bin/python nvidia-ml-py 2>/dev/null || /venv/main/bin/python -m pip install -q nvidia-ml-py
step "8/8 check: every weight of the manifest present, torch sees the GPU, the CUDA llama.cpp offloads"
/venv/dabai/bin/python - <<'PY'
import json, pathlib, sys
m = json.loads(pathlib.Path("reports/experiments/r13-spec/IMAGE/models_manifest.json").read_text()); bad = []
for r in m:
    if r["status"].startswith(("drop", "OPTIONAL")) or r["path"].startswith(("pip", "models/hf")): continue
    p = pathlib.Path(r["path"]) if r["path"].startswith("/") else pathlib.Path(r["path"])
    if not p.exists(): bad.append(r["name"])
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
sys.path.insert(0, "third_party/llama_cuda"); import llama_cpp; print("llama_cpp", llama_cpp.__version__, "gpu offload", llama_cpp.llama_supports_gpu_offload())
print("MISSING:", bad) if bad else print("all weights present")
PY
echo "SETUP_FROM_SCRATCH_DONE  (smoke test: ID=t03 bash scripts/r9/e2e.sh /path/to/03.mp4 /tmp/t03)"
