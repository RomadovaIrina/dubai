#!/bin/bash
# Final assembly on CPU: restored video + audio -> H.264/AAC mp4 (NVENC is non-functional on this host).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
V="${DUB_ROOT}/reports/codeformer_demo.mp4"; A="${DUB_ROOT}/third_party/latentsync/assets/demo1_audio.wav"; OUT="${DUB_ROOT}/reports/final_demo.mp4"
[ -s "$V" ] || V="${DUB_ROOT}/reports/latentsync_demo.mp4"
ffmpeg -y -hide_banner -loglevel error -i "$V" -i "$A" -map 0:v:0 -map 1:a:0 -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -c:a aac -b:a 192k -shortest "$OUT"
ffprobe -v error -show_entries stream=codec_name,width,height -of csv=p=0 "$OUT"
echo "PASS ffmpeg mux -> $(basename "$OUT")"
