#!/bin/bash
# r13 old03 2-panel: v1 (left) | fixed (right), full length <= 25 MB, + a 0.5x face crop around the worst colour jumps (86.5-91 s: the guard swap + the KeySync end)
#   r13_panel.sh <fixed.mp4> <out prefix>
V1=/workspace/dub/outbox/r13_ks/v1check/old03_v1_KeySync.mp4; FX=$1; P=$2
LB="drawtext=font=DejaVu Sans:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6:x=12:y=12:text"
D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $FX); BR=$(python3 -c "print(int(min(8000, 25e6 * 8 / $D * 0.92 / 1000 - 160)))")
ffmpeg -v error -y -i $V1 -i $FX -filter_complex "[0:v]$LB='v1 (now)'[a];[1:v]$LB='colour fix'[b];[a][b]hstack=2[v]" -map "[v]" -map 1:a \
  -c:v libx264 -preset slow -crf 20 -maxrate ${BR}k -bufsize $((2 * BR))k -pix_fmt yuv420p -c:a copy -movflags +faststart ${P}_2panel.mp4
ffmpeg -v error -y -ss 86.5 -t 4.5 -i $V1 -ss 86.5 -t 4.5 -i $FX -filter_complex \
  "[0:v]crop=340:340:92:272,scale=480:480:flags=lanczos,$LB='v1 0.5x'[a];[1:v]crop=340:340:92:272,scale=480:480:flags=lanczos,$LB='colour fix 0.5x'[b];[a][b]hstack=2,setpts=2*PTS[v];[1:a]atempo=0.5[au]" \
  -map "[v]" -map "[au]" -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart ${P}_jump_86-91s_halfspeed.mp4
ls -la ${P}_2panel.mp4 ${P}_jump_86-91s_halfspeed.mp4
