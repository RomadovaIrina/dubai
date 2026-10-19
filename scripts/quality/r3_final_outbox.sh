#!/bin/bash
# Round 3 D: publish the fixed finals + before / after clips (user defects 19.10: torn lips on single 30 fps frames, abrupt phrase cuts).
#   outbox/<id>_r3final_fixed.mp4                         A R3 generator + B fps master + B final composite + flow_safe retime, audio-r4 (fixed cuts, -16 LUFS)
#   outbox/<id>_<t0>-<t1>s_BEFORE-AFTER.mp4 (+ _half)     BEFORE = <id>_r3cand_audio-r3_final (r3cand, audio-r3b) | AFTER = r3final_fixed; sound = AFTER
#   outbox/<id>_<t0>-<t1>s_before_then_after.mp4          the same window twice in a row (BEFORE, then AFTER), each with its own sound
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
V=/workspace/dub/artifacts/r3d/video; OB=/workspace/dub/outbox; F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
for id in 01 02 03 04 05; do [ -f $V/${id}_r3final_fix30.mp4 ] && cp $V/${id}_r3final_fix30.mp4 $OB/${id}_r3final_fixed.mp4; done
clip() {   # id t0 t1
  id=$1; t0=$2; t1=$3; d=$(python -c "print(round($t1-$t0,3))"); B=$OB/${id}_r3cand_audio-r3_final.mp4; A=$OB/${id}_r3final_fixed.mp4; n=$OB/${id}_${t0}-${t1}s
  [ -f $B ] && [ -f $A ] || { echo "skip clip $id"; return; }
  ffmpeg -loglevel error -y -ss $t0 -t $d -i $B -ss $t0 -t $d -i $A -filter_complex \
    "[0:v]drawtext=fontfile=$F:text='BEFORE (r3cand)':x=10:y=10:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.6[a];[1:v]drawtext=fontfile=$F:text='AFTER (fixed)':x=10:y=10:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.6[b];[a][b]hstack=inputs=2[v]" \
    -map "[v]" -map 1:a:0 -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a aac -b:a 192k ${n}_BEFORE-AFTER.mp4
  ffmpeg -loglevel error -y -i ${n}_BEFORE-AFTER.mp4 -filter_complex "[0:v]setpts=2*PTS[v];[0:a]atempo=0.5[a]" -map "[v]" -map "[a]" -c:v libx264 -crf 16 -pix_fmt yuv420p ${n}_BEFORE-AFTER_half.mp4
  ffmpeg -loglevel error -y -ss $t0 -t $d -i $B -ss $t0 -t $d -i $A -filter_complex \
    "[0:v]drawtext=fontfile=$F:text='BEFORE':x=10:y=10:fontsize=28:fontcolor=yellow:box=1:boxcolor=black@0.6[a];[1:v]drawtext=fontfile=$F:text='AFTER':x=10:y=10:fontsize=28:fontcolor=yellow:box=1:boxcolor=black@0.6[b];[a][0:a][b][1:a]concat=n=2:v=1:a=1[v][au]" \
    -map "[v]" -map "[au]" -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a aac -b:a 192k ${n}_before_then_after.mp4
  echo "== clips $id $t0-$t1"
}
clip 04 20 25
clip 03 ${CLIP03:-35 40}
