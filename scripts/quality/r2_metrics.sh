#!/bin/bash
# r2 timing metrics for one video: LS reference mask from the cand5 run (face_ref.py), then dub_gaps.py + pause_rhythm.py on each tagged manifest.
#   bash scripts/quality/r2_metrics.sh 03 OUT cand5=/path/cand5.manifest.json r2=/path/r2.manifest.json ...
set -euo pipefail
cd "$(dirname "$0")/../.."; . scripts/env.sh; export DUB_ROOT=$(pwd)
id=$1; OUT=$2; shift 2; mkdir -p "$OUT/face" "$OUT/gaps" "$OUT/pauses"
ref=""; for kv in "$@"; do t=${kv%%=*}; m=${kv#*=}; [ -z "$ref" ] && { ref="$OUT/face/${id}.ref.manifest.json"; [ -f "$ref" ] || python scripts/quality/face_ref.py --manifest "$m" --out "$ref"; }
  python scripts/quality/dub_gaps.py --manifest "$m" --ref-manifest "$ref" --tag "$t" --out "$OUT/gaps" | head -1
  python scripts/quality/pause_rhythm.py --manifest "$m" --tag "$t" --out "$OUT/pauses" > /dev/null
done
