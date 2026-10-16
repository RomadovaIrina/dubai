#!/bin/bash
# cand5 baseline (reports/quality/feedback_2026-10-13/README.md "new candidate"), one PILOT_REFRESH_ROOT per video.
set -u
cd /workspace/dub/dubai
export DUB_ROOT=/workspace/dub/dubai
CAND5="--phrase-cut acoustic --long-pause-func 1.2 --cut-fade adaptive --tts-gap-qa on --gate-merge-gap-s 1.2 --mouth-only-pasteback on --codeformer-scope mouth --mask-smooth 3 --ls-landmark-smooth 2"
for id in "$@"; do
  R=/workspace/dub/artifacts/baseline/$id; mkdir -p "$R"
  echo "== $id start $(date -Iseconds)"
  s=$(date +%s)
  PILOT_REFRESH_ROOT="$R" DUB_EXTRA_ARGS="$CAND5" bash scripts/pilot/refresh_e2e.sh "$id" > "$R/refresh.log" 2>&1
  echo "== $id done rc=$? wall=$(( $(date +%s) - s ))s $(date -Iseconds)"
  tail -3 "$R/refresh.log"
done
