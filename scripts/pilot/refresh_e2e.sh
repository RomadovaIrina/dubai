#!/bin/bash
# Pipeline refresh (2026-10-02): the five CURRENT production-candidate E2E outputs (01-05) with the plain runner command
#   run_clean_pipeline_05.py --target-lang en --alignment burst --video-backend optimized --codeformer optimized
# (all other settings = runner defaults: TTS fp16 token-ID guard, natural chunking, TTS QA/retry, duration-aware
# retranslation, atempo cap 1.15 / hard cap 1.2, burst alignment, optimized LatentSync, optimized CodeFormer w 1.0 b8 fp16).
# One nvidia-smi sampler (1 s) per run -> <out>.nvsmi.log (device-wide peak VRAM); runs are strictly sequential.
#   bash scripts/pilot/refresh_e2e.sh            # 01 02 03 04 05
#   bash scripts/pilot/refresh_e2e.sh 03         # one video
set -uo pipefail
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; . "${DUB_ROOT}/scripts/env.sh"; cd "${DUB_ROOT}"
R="${PILOT_REFRESH_ROOT:-/tmp/pilot_refresh}"; F="$R/final"; W="$R/work"; L="$R/logs"; mkdir -p "$F" "$W" "$L"
IDS=("$@"); [ ${#IDS[@]} -eq 0 ] && IDS=(01 02 03 04 05)
for id in "${IDS[@]}"; do
  in=$(ls test_videos/"${id}".* | head -1); out="$F/${id}_final.mp4"
  echo "===== $id start $(date -Iseconds) input=$in"
  echo "--- nvidia-smi before $id ---"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -l 1 > "$F/${id}_final.nvsmi.log" 2>/dev/null & NV=$!
  s=$(date +%s)
  python scripts/pilot/run_clean_pipeline_05.py --input "$in" --output "$out" --target-lang en \
      --alignment burst --video-backend optimized --codeformer optimized --work-dir "$W" > "$L/${id}_e2e.log" 2>&1; rc=$?
  kill $NV 2>/dev/null; wait $NV 2>/dev/null
  peak=$(awk -F', ' '{print $2}' "$F/${id}_final.nvsmi.log" | sort -n | tail -1)
  echo "===== $id exit $rc wall $(( $(date +%s) - s ))s peak_nvsmi_MiB=$peak"
  tail -4 "$L/${id}_e2e.log"
done
