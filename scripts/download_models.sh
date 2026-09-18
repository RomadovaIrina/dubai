#!/bin/bash
# Downloads every weight the pilot needs into models/ (idempotent; re-run to resume). Qwen: Q8_0 since the pilot 0.9 decision (2026-09-18).
# Gated pyannote models are pulled lazily by check_pyannote.py using HF_TOKEN from /workspace/.env.
set -euo pipefail
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/venv/dabai/bin/python}"
M="${DUB_ROOT}/models"; mkdir -p "$M"
export HF_HOME="${HF_HOME:-$M/hf}" HF_HUB_DISABLE_TELEMETRY=1
[ -f /workspace/.env ] && set -a && . /workspace/.env && set +a

snap(){ # snap <repo> <local_dir> [allow_patterns...]
  local repo="$1" dir="$2"; shift 2
  "${PY}" - "$repo" "$dir" "$@" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
repo, d, *pat = sys.argv[1:]
p = snapshot_download(repo_id=repo, local_dir=d, allow_patterns=pat or None)
print("ok", repo, "->", p)
PYEOF
}

echo "== faster-whisper large-v3 (~3.1 GB)"; snap Systran/faster-whisper-large-v3 "$M/whisper-large-v3"
echo "== Qwen2.5-7B-Instruct GGUF Q8_0 (~8.1 GB, 3 shards; production quant per pilot 0.9)"; snap Qwen/Qwen2.5-7B-Instruct-GGUF "$M/qwen2.5-7b-instruct-gguf" "qwen2.5-7b-instruct-q8_0*.gguf"
echo "== Chatterbox multilingual v3"; snap ResembleAI/chatterbox "$M/chatterbox" "ve.pt" "s3gen.pt" "conds.pt" "t3_mtl23ls_v3.safetensors" "grapheme_mtl_merged_expanded_v1.json" "Cangjie5_TC.json" "*.json"
echo "== LatentSync 1.6 (unet 5 GB + whisper tiny)"; snap ByteDance/LatentSync-1.6 "$M/latentsync" "latentsync_unet.pt" "whisper/tiny.pt"
echo "== sd-vae-ft-mse (LatentSync VAE)"; snap stabilityai/sd-vae-ft-mse "$M/sd-vae-ft-mse" "*.json" "*.safetensors"
# LatentSync expects ./checkpoints/{latentsync_unet.pt,whisper/tiny.pt} relative to its repo root
LS="${DUB_ROOT}/third_party/latentsync/checkpoints"; mkdir -p "$LS"
ln -sfn "$M/latentsync/latentsync_unet.pt" "$LS/latentsync_unet.pt"; ln -sfn "$M/latentsync/whisper" "$LS/whisper"
# CodeFormer + facelib weights (github releases) into the clone's weights/ dir
CF="${DUB_ROOT}/third_party/CodeFormer"; mkdir -p "$CF/weights/CodeFormer" "$CF/weights/facelib"
dl(){ [ -s "$2" ] || curl -fL --retry 3 -o "$2" "$1"; echo "ok $2"; }
dl https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth "$CF/weights/CodeFormer/codeformer.pth"
dl https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth "$CF/weights/facelib/detection_Resnet50_Final.pth"
dl https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth "$CF/weights/facelib/parsing_parsenet.pth"
# facexlib (RetinaFace) uses the identical detector weights; pre-seed its cache so first use is offline
FX="$M/facexlib"; mkdir -p "$FX"; ln -sfn "$CF/weights/facelib/detection_Resnet50_Final.pth" "$FX/detection_Resnet50_Final.pth"
echo "== done"; du -sh "$M"/* 2>/dev/null
