#!/bin/bash
# LatentSync 1.6 inference on its demo video + demo audio (fp16 auto on sm_120, ~18 GB VRAM).
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "${DUB_ROOT}/third_party/latentsync"
OUT="${DUB_ROOT}/reports/latentsync_demo.mp4"
AUDIO="${LS_AUDIO:-assets/demo1_audio.wav}"
python -m scripts.inference --unet_config_path configs/unet/stage2_512.yaml --inference_ckpt_path checkpoints/latentsync_unet.pt \
  --inference_steps 20 --guidance_scale 1.5 --enable_deepcache --seed 1247 \
  --video_path assets/demo1_video.mp4 --audio_path "$AUDIO" --video_out_path "$OUT"
[ -s "$OUT" ] || { echo "FAIL: no output"; exit 1; }
ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT" | xargs -I{} echo "output duration {} s -> $(basename "$OUT")"
echo "PASS latentsync"
