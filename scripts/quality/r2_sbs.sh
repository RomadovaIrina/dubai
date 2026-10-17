#!/bin/bash
# Round 2 track B: labelled side-by-side clips (any frame rate, time-based): full frame + 2x nose / mouth crop, normal + half speed.
#   bash scripts/quality/r2_sbs.sh <out_prefix> <t0> <dur> <crop x,y,w,h | none> LABEL=file [LABEL=file ...]   (audio from the 2nd panel)
set -eu
out=$1; t0=$2; dur=$3; crop=$4; shift 4
IFS=, read cx cy cw ch <<< "${crop/none/0,0,8,8}"; ins=(); lab=(); i=0; f1=""; f2=""
for p in "$@"; do lab+=("${p%%=*}"); ins+=(-ss "$t0" -t "$dur" -i "${p#*=}"); done
n=${#lab[@]}
for ((i=0;i<n;i++)); do
  f1+="[$i:v]setpts=PTS-STARTPTS,drawtext=text='${lab[$i]}':x=6:y=6:fontsize=22:fontcolor=yellow:box=1:boxcolor=black@0.6[a$i];"
  f2+="[$i:v]setpts=PTS-STARTPTS,crop=$cw:$ch:$cx:$cy,scale=iw*2:ih*2:flags=lanczos,drawtext=text='${lab[$i]} 2x':x=6:y=6:fontsize=22:fontcolor=yellow:box=1:boxcolor=black@0.6[c$i];"
done
s1=""; s2=""; for ((i=0;i<n;i++)); do s1+="[a$i]"; s2+="[c$i]"; done
mkdir -p "$(dirname $out)"
ffmpeg -loglevel error -y "${ins[@]}" -filter_complex "${f1}${s1}hstack=inputs=$n[v]" -map "[v]" -map 1:a:0 -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a aac -shortest ${out}_full.mp4
kinds="full"
if [ "$crop" != none ]; then
  ffmpeg -loglevel error -y "${ins[@]}" -filter_complex "${f2}${s2}hstack=inputs=$n[v]" -map "[v]" -map 1:a:0 -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a aac -shortest ${out}_crop2x.mp4; kinds="full crop2x"
fi
for k in $kinds; do ffmpeg -loglevel error -y -i ${out}_$k.mp4 -filter_complex "[0:v]setpts=2*PTS[v];[0:a]atempo=0.5[a]" -map "[v]" -map "[a]" -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a aac ${out}_${k}_half.mp4; done
ls -la ${out}_*
