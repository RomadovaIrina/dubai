#!/bin/bash
# WATCH with N panels (artifact tags), full video, demucs mix:  bash watchN.sh <id> <name> <tag>:<label> ...
id=$1; name=$2; shift 2; B=/workspace/dub/artifacts/r4m; O=/workspace/dub/outbox/r9fx/watch; mkdir -p $O; in=(); fl=""; k=0; v=""
for spec in "$@"; do T=${spec%%:*}; L=${spec#*:}; in+=(-i $B/$T/${id}_$T.mp4); fl="$fl[$k:v]scale=-2:800,drawtext=text='$L':x=8:y=8:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6[v$k];"; v="$v[v$k]"; k=$((k + 1)); done
ffmpeg -loglevel error -y "${in[@]}" -i /workspace/dub/r9/runs/$id/${id}_DabAI_r9_demucs.mp4 -filter_complex "${fl}${v}hstack=$k[v]" -map "[v]" -map $k:a:0 -c:v libx264 -crf 20 -pix_fmt yuv420p -c:a copy -shortest $O/${id}_WATCH_$name.mp4 && echo "== $O/${id}_WATCH_$name.mp4 $(du -m $O/${id}_WATCH_$name.mp4 | cut -f1) MB"
