#!/bin/bash
# pngs.sh LIST OUTDIR [SET]  (LIST lines: id|t|png|label)
L=$1; O=$2; SET=${3:-now}; Q=/workspace/dub/q; R=/workspace/dub/r16in/src; QS=$Q/src; mkdir -p $O
declare -A SRC=([m03]=$QS/story-1774249919482-367803614.MOV [m18]=$QS/story-1774250817106-616905094.mp4 [old03]=$R/old03.mp4 [q01]=$R/q01_amir.mp4 [q01_subs]=$R/q01_amir.mp4 [q02]=$R/q02_vera.mp4 [q04]=$R/q04_farikh.mp4 [q06]=$R/q06_suvorov.mp4)
while IFS='|' read id t png label; do [ -n "$id" ] || continue
  /workspace/dub/qenv/bin/python $Q/scripts/r16q/moment_png.py --final /workspace/dub/r16in/day3/now_1011/now_$id.mp4 --src ${SRC[$id]} --npz $Q/out/$SET/$id.det.npz --t $t --step 2 --half 6 --y0 ${Y0:-0.30} --y1 1.0 --tile 260 --out $O/$png --label "$label" 2>/dev/null < /dev/null && echo ok $png
done < $L
