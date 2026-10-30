#!/bin/bash
# WATCH: r9 | r9fx side by side, full video, demucs mix   (bash watch2.sh <id> ...)
O=/workspace/dub/outbox/r9fx/watch; mkdir -p $O
for id in "$@"; do a=/workspace/dub/r9/runs/$id/${id}_DabAI_r9_demucs.mp4; b=/workspace/dub/outbox/r9fx/${id}_DabAI_r9fx_demucs.mp4
  [ -f $b ] || { echo "no $b"; continue; }
  ffmpeg -loglevel error -y -i $a -i $b -filter_complex "[0:v]scale=-2:960,drawtext=text='r9 (shipped)':x=10:y=10:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6[x];[1:v]scale=-2:960,drawtext=text='r9fx (fix)':x=10:y=10:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6[y];[x][y]hstack" -map 0:a:0 -c:v libx264 -crf 20 -pix_fmt yuv420p -c:a copy $O/${id}_WATCH_r9_vs_r9fx.mp4 && echo "== $O/${id}_WATCH_r9_vs_r9fx.mp4 $(du -m $O/${id}_WATCH_r9_vs_r9fx.mp4 | cut -f1) MB"
done
