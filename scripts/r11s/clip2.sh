#!/bin/bash
# r11 SYNC: full-length 2-panel user clip  LEFT = r9 + fix (now) | RIGHT = an arm, demucs mix, H.264, <= MAXMB (default 25).
#   bash scripts/r11s/clip2.sh <id> <arm tag> "<right label>" [out dir]
id=$1; T=$2; RL=$3; OD=${4:-/workspace/dub/outbox/r11_sync}; MAXMB=${MAXMB:-25}; mkdir -p $OD
B=/workspace/dub/artifacts/r4m; L=$B/r9f_$id/${id}_r9f_$id.mp4; R=$B/r9g_$id/work_${id}_r9g_$id/${id}_r9g_${id}_25fps_$T.mp4; A=/workspace/dub/r10/runs/$id/${id}_DabAI_r9_demucs.mp4
F=$(fc-list 2>/dev/null | grep -i "DejaVuSans-Bold.ttf" | head -1 | cut -d: -f1); FF=""; [ -n "$F" ] && FF="fontfile=$F:"
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $A); kb=$(python3 -c "print(int($MAXMB*8*1024*0.92/$dur - 160))")
O=$OD/${id}_r9fix_vs_sync_$T.mp4
DT="x=12:y=12:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.6"
ffmpeg -loglevel error -y -i $L -i $R -i $A -filter_complex \
  "[0:v]scale=-2:960,setsar=1,drawtext=${FF}text='r9+fix (now)':$DT[l];[1:v]scale=-2:960,setsar=1,drawtext=${FF}text='$RL':$DT[r];[l][r]hstack=inputs=2[v]" \
  -map "[v]" -map 2:a:0 -c:v libx264 -preset slow -b:v ${kb}k -maxrate $((kb * 2))k -bufsize $((kb * 2))k -pix_fmt yuv420p -c:a aac -b:a 128k -shortest -movflags +faststart $O
echo "== clip $O $(du -m $O | cut -f1) MB ${dur}s"
