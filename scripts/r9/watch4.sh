#!/bin/bash
# WATCH: full video, 4 panels at half size, the demucs mix: r9 shipped | r9 + fx | shift16 + fx | fixed + fx  (fx = layer repair + inner-mouth detail 0.4)
id=$1; B=/workspace/dub/artifacts/r4m; O=/workspace/dub/outbox/r9_e2e/watch_mouth; mkdir -p $O
P=(r9f_$id r9f_${id}fx r9f_${id}shift16L05fx r9f_${id}fixedL05fx); L=("r9 shipped" "r9 + fix" "ref shift16 + fix" "ref fixed + fix")
in=(); fl=""; for k in 0 1 2 3; do in+=(-i $B/${P[$k]}/${id}_${P[$k]}.mp4); fl="$fl[$k:v]scale=-2:640,drawtext=text='${L[$k]}':x=8:y=8:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[v$k];"; done
ffmpeg -loglevel error -y "${in[@]}" -i /workspace/dub/r9/runs/$id/${id}_DabAI_r9_demucs.mp4 -filter_complex "${fl}[v0][v1][v2][v3]hstack=4[v]" -map "[v]" -map 4:a:0 -c:v libx264 -crf 20 -preset medium -pix_fmt yuv420p -c:a copy -shortest $O/${id}_WATCH_mouth_r9_fix_shift16_fixed.mp4 && echo "== $O/${id}_WATCH_mouth_r9_fix_shift16_fixed.mp4 $(du -m $O/${id}_WATCH_mouth_r9_fix_shift16_fixed.mp4 | cut -f1) MB"
