#!/bin/bash
# r15 K: 0.5x mouth 2-panels production KeySync | <arm> at the user eye spots (old03 3-7 s, m03 17-21 s, m18 2-6 s) -> outbox/r15_k/
# usage: bash k_clips.sh <arm> [<label>]
A=$1; LB=${2:-$1}; K=/workspace/dub/keysync; O=/workspace/dub/outbox/r15_k; mkdir -p $O; cd $K
for s in "q18 m18 2-6" "m03 m03 17-21" "qo3 old03 3-7"; do set -- $s
  [ -s fast/$A/eval_$1/paste.mp4 ] || { echo "no paste $A $1"; continue; }
  bash k_clip.sh $1 fast/prod/eval_$1/paste.mp4 "prod KS" fast/$A/eval_$1/paste.mp4 "$LB" $3 $O/${2}_mouth_prodKS_vs_${A}_${3}s_half_speed.mp4
done
