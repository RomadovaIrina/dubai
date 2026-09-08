#!/bin/bash
# CodeFormer face restoration on the LatentSync output (video in -> video out), w=0.5, no upscale.
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "${DUB_ROOT}/third_party/CodeFormer"
IN="${CF_INPUT:-${DUB_ROOT}/reports/latentsync_demo.mp4}"
[ -s "$IN" ] || { echo "FAIL: input $IN missing (run check_latentsync.sh first)"; exit 1; }
OUTDIR="${DUB_ROOT}/reports/codeformer"
python inference_codeformer.py -i "$IN" -o "$OUTDIR" -w 0.5 -s 1 --detection_model retinaface_resnet50 --save_video_fps 25
OUT=$(ls -t "$OUTDIR"/*.mp4 | head -1); [ -s "$OUT" ] || { echo "FAIL: no output video"; exit 1; }
cp "$OUT" "${DUB_ROOT}/reports/codeformer_demo.mp4"; echo "-> reports/codeformer_demo.mp4"
echo "PASS codeformer"
