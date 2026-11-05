#!/bin/bash
# r15 W: the eye clips of two arms on one held-out video -> /workspace/dub/outbox/r15_w/:
#   <id>_full_<L>_vs_<R>.mp4               side by side, full length, 720 px high, the demucs mix
#   <id>_mouth_<L>_vs_<R>_spots_half_speed.mp4   the mouth 2-panel (scripts/r13/distill/mouth_2panel.py) at the user's eye spots, 0.5x, time burnt in
#   bash scripts/r15/pair.sh <old03|m03|m18> <L arm> <R arm> ["L label" "R label"]
set -u
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
X=/workspace/dub/r15w/eval; OB=/workspace/dub/outbox/r15_w; mkdir -p $OB; id=$1; L=$2; R=$3; LL=${4:-$L}; RL=${5:-$R}
declare -A SP=([old03]="2.75:4.25 4.75:6.25 20.55:21.25 23.5:24.5 26.75:28.25" [m03]="16.5:21.5" [m18]="1.5:6.5")
vid(){ [ "$1" = q2 ] && echo $X/q2_$id/q2_$id.mp4 || echo $X/${1}_$id/${1}_$id.mp4; }
lv=$(vid $L); rv=$(vid $R); LM=$(find /workspace/dub/artifacts/r4m -path "*r9g_q2_${id}*/*" -name "q2_${id}_source_lm_F.npz" | head -1); T=$(mktemp -d)
ffmpeg -v error -y -i $lv -i $rv -filter_complex "[0:v]scale=-2:720,drawtext=text='$LL':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6[l];[1:v]scale=-2:720,drawtext=text='$RL':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6[r];[l][r]hstack=inputs=2[v]" \
  -map "[v]" -map 0:a -c:v libx264 -crf 20 -preset fast -c:a aac $OB/${id}_full_${L}_vs_${R}.mp4
python scripts/r13/distill/mouth_2panel.py --left $lv --right $rv --lm $LM --out $T/m.mp4 --labels "$LL" "$RL" > /dev/null
parts=""; k=0
for span in ${SP[$id]}; do s=${span%:*}; e=${span#*:}; k=$((k+1))
  ffmpeg -v error -y -ss $s -to $e -i $T/m.mp4 -filter_complex "[0:v]setpts=2*PTS,drawtext=text='t=$s-$e s (0.5x)':x=(w-tw)/2:y=h-30:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[v];[0:a]atempo=0.5[a]" \
    -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset fast -c:a aac -ar 48000 $T/ts_$k.mp4 && parts="$parts -i $T/ts_$k.mp4"; done
n=$(echo $parts | grep -o "\-i" | wc -l)
ffmpeg -v error -y $parts -filter_complex "$(for i in $(seq 0 $((n-1))); do echo -n "[$i:v][$i:a]"; done)concat=n=$n:v=1:a=1[v][a]" -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset fast -c:a aac $OB/${id}_mouth_${L}_vs_${R}_spots_half_speed.mp4
cp $T/m.mp4 $OB/${id}_mouth_${L}_vs_${R}.mp4; rm -rf $T
ls -la $OB/${id}_*${L}_vs_${R}* | awk '{print $5, $9}'
