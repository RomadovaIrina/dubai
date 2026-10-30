#!/bin/bash
# Agent C r3 (2026-10-19, server A): audio-only run of the clean pipeline for ONE video with the r2 audio defaults (--audio-policy r2 = cand5 +
# flow 0.2 + spread pauses + slow-down floor 0.875, Spleeter 2stems bed, declick, duck -4 dB / 250 ms / 600 ms, residue duck -15 dB).
# GPU shared with agent A (coordinator 19.10: <= ~10 GB VRAM for C). Everything on the GPU: Chatterbox + QA, Qwen (CUDA llama-cpp-python
# 0.3.35 built for sm_120 into /workspace/dub/c/pylib-llama-cuda, put in front of the shared CPU build via PYTHONPATH), Spleeter (torch port).
#   scripts/quality/r3_audio_run.sh 03 [tag] [extra runner flags...]      (QWEN_GPU=0: Qwen on the CPU as in the first audio-r2 run)
set -euo pipefail
id=$1; tag=${2:-r2}; shift $(( $# >= 2 ? 2 : 1 ))
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS=/workspace/dub/c/runs/$tag; mkdir -p "$RUNS"
src=$(ls ${DUB_INPUTS:-/workspace/dub/inputs/test_videos_raw}/${id}.* | head -1)
source /venv/dabai/bin/activate
set -a; . /workspace/.env 2>/dev/null || true; set +a          # HF token for the gated pyannote pipeline (never printed)
export DUB_ROOT="$REPO" DUB_SPLEETER_PY=/workspace/dub/c/venv-spleeter/bin/python DUB_THREADS=${DUB_THREADS:-8}
qwen=()
if [ "${QWEN_GPU:-1}" = 1 ]; then export PYTHONPATH=/workspace/dub/c/pylib-llama-cuda${PYTHONPATH:+:$PYTHONPATH}; qwen=(--qwen-gpu-layers -1); fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
python scripts/pilot/run_clean_pipeline_05.py --input "$src" --output "$RUNS/${id}_${tag}.mp4" --target-lang en \
  --alignment burst --stop-after audio --no-codeformer --face-stack mediapipe --audio-policy r2 "${qwen[@]}" --work-dir "$RUNS" "$@" 2>&1 | tee "$RUNS/${id}.log"
