#!/bin/bash
# Phase 1: production baseline (reports/pilot/FINAL_ML_CONFIG.md canonical command, unchanged) on the test videos, once each.
# Outputs + manifests -> /tmp/dabai_quality/baseline/, runtime media -> /tmp/dabai_quality/work/work_<id>/ (kept for audio/timing analysis).
# Usage: scripts/quality/run_baseline.sh 04 03 01 02 05
set -uo pipefail
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; . "${DUB_ROOT}/scripts/env.sh"; cd "${DUB_ROOT}"
Q=/tmp/dabai_quality; OUT="${Q}/baseline"; WORK="${Q}/work"; mkdir -p "$OUT" "$WORK"
for id in "$@"; do
  src=$(ls test_videos/${id}.mp4 test_videos/${id}.MP4 2>/dev/null | head -1)
  [ -n "$src" ] || { echo "===== $id MISSING"; continue; }
  echo "===== $id start $(date -Iseconds) src=$src"
  s=$(date +%s)
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > "${OUT}/${id}_baseline.nvsmi.log" 2>/dev/null & NV=$!
  python scripts/pilot/run_clean_pipeline_05.py --input "$src" --output "${OUT}/${id}_baseline.mp4" --target-lang en --no-codeformer \
    --work-dir "$WORK" \
    --video-backend optimized --face-router retinaface --face-min-w 50 --face-min-h 80 --retina-conf 0.8 --router-stride 3 \
    --speech-gate on --gate-margin-s 0.24 --gate-merge-gap-s 0.6 --crossfade-frames 4 --deepcache-interval 5 \
    --window-batch-size 2 --sdpa-backend auto --compile-backend none --seed 1247 > "${OUT}/${id}_baseline.log" 2>&1
  rc=$?; kill $NV 2>/dev/null; wait $NV 2>/dev/null
  echo "===== $id exit $rc wall $(( $(date +%s) - s ))s peak_nvsmi_MiB=$(sort -n "${OUT}/${id}_baseline.nvsmi.log" | tail -1)"
done
echo "===== all done"
