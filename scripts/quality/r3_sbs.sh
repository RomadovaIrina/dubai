#!/bin/bash
# Round 3 D: labelled side by side, any frame rates (every panel resampled to the first panel's fps, nearest frame), audio from AUDIO_PANEL (default last).
#   bash scripts/quality/r3_sbs.sh <out.mp4> [-ss T -t D] LABEL=file LABEL=file ...      HALF=1 -> also <out>_half.mp4 (0.5x, audio dropped)
set -u
out=$1; shift; trim=""; if [ "${1:-}" = "-ss" ]; then trim="-ss $2 -t $4"; shift 4; fi
F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf; ins=(); fil=""; n=0; labs=()
fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0 "${1#*=}")
for p in "$@"; do lab=${p%%=*}; f=${p#*=}; ins+=($trim -i "$f")
  fil+="[$n:v]fps=$fps,scale=-2:848,setsar=1,drawtext=fontfile=$F:text='$lab':x=10:y=10:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.6[v$n];"; n=$((n+1)); done
st=""; for i in $(seq 0 $((n-1))); do st+="[v$i]"; done
ap=${AUDIO_PANEL:-$((n-1))}
ffmpeg -loglevel error -y "${ins[@]}" -filter_complex "${fil}${st}hstack=inputs=$n[v]" -map "[v]" -map $ap:a:0 -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -b:a 160k -shortest "$out"
[ "${HALF:-0}" = 1 ] && ffmpeg -loglevel error -y -i "$out" -vf "setpts=2*PTS" -an -c:v libx264 -crf 18 -pix_fmt yuv420p "${out%.mp4}_half.mp4"
echo "$out"
