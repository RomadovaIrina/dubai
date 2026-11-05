#!/bin/bash
# r15: the student arm vs the r14 student s4p1500c8 (2-panel full + 0.5x mouth), and the eye-check spans (m18 2-6 s, m03 17-21 s; 0.5x, time burnt in)
# against both q2 + F3 and s4p1500c8 -> outbox/r15_d/.   bash scripts/r14/r15_clips.sh <TAG> [REF_TAG=s4p1500c8]
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
TAG=${1:?TAG}; R=${2:-s4p1500c8}; X=${EVAL_DIR:-/workspace/dub/r14d/eval}; OB=${OB:-/workspace/dub/outbox/r15_d}; mkdir -p $OB
declare -A SPAN=([m18]=2:6 [m03]=17:21)
for id in old03 m03 m18; do
  lm=$(find /workspace/dub/artifacts -path "*${Q2:-q2}_$id*" -name "*source_lm_F.npz" 2>/dev/null | head -1); [ -n "$lm" ] || { echo "no landmarks for $id"; continue; }
  v=$X/${TAG}_$id/${TAG}_$id.mp4; r=$X/${R}_$id/${R}_$id.mp4; { [ -s $v ] && [ -s $r ]; } || { echo "missing $v or $r"; continue; }
  python scripts/r13/distill/mouth_2panel.py --left $r --right $v --lm $lm --out $OB/${id}_mouth_${R}_vs_${TAG}.mp4 --labels "student $R (r14)" "student $TAG (r15)"
  ffmpeg -v error -y -i $OB/${id}_mouth_${R}_vs_${TAG}.mp4 -filter_complex "[0:v]setpts=2*PTS[v];[0:a]atempo=0.5[a]" -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset fast $OB/${id}_mouth_${R}_vs_${TAG}_half_speed.mp4
  ffmpeg -v error -y -i $r -i $v -filter_complex "[0:v]scale=-2:720[l];[1:v]scale=-2:720[r];[l][r]hstack=inputs=2[v]" -map "[v]" -map 0:a -c:v libx264 -crf 20 -preset fast -c:a aac $OB/${id}_full_${R}_vs_${TAG}.mp4
  sp=${SPAN[$id]:-}; [ -n "$sp" ] || continue; s=${sp%:*}; e=${sp#*:}
  for L in ${Q2:-q2} $R; do   # the span cut from each 2-panel (left = L, right = TAG), mouth and full, at 0.5x
    for kind in mouth full; do
      src=$OB/${id}_${kind}_${L}_vs_${TAG}.mp4; [ -s $src ] || continue
      ffmpeg -v error -y -ss $s -to $e -i $src -filter_complex "[0:v]setpts=2*PTS,drawtext=text='t=$s-$e s (0.5x)':x=(w-tw)/2:y=h-30:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[v];[0:a]atempo=0.5[a]" \
        -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset fast -c:a aac $OB/${id}_${kind}_${L}_vs_${TAG}_t${s}-${e}_half_speed.mp4
    done
  done
done
ls $OB; echo CLIPS_DONE
