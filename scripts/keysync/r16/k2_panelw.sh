#!/bin/bash
# r16 K2: k2_panelw.sh OUT V W0 T0 T1 label=window.mp4 ... : panel of WINDOW renders (window starts at W0 s), time burnt in as the video time, 0.5x, the final mix audio
out=$1; v=$2; w0=$3; t0=$4; t1=$5; shift 5; F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf; P=/workspace/dub/r16in/HANDOFF_K/paste_inputs
a0=$(python3 -c "print($t0-$w0)"); a1=$(python3 -c "print($t1-$w0)"); ins=""; fc=""; i=0; st=""
for lv in "$@"; do l=${lv%%=*}; f=${lv#*=}; ins="$ins -ss $a0 -to $a1 -i $f"
  fc="$fc[$i:v]setpts=PTS-STARTPTS,drawtext=fontfile=$F:text='$l':x=8:y=8:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.5,drawtext=fontfile=$F:text='%{pts\:hms\:$t0}':x=8:y=h-34:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.5,setpts=2*PTS[v$i];"; st="$st[v$i]"; i=$((i+1)); done
ffmpeg -v error -y $ins -ss $t0 -to $t1 -i $P/$v/final_mix_aac.m4a -filter_complex "${fc}${st}hstack=inputs=$i[v];[$i:a]asetpts=PTS-STARTPTS,atempo=0.5[a]" -map "[v]" -map "[a]" -c:v libx264 -crf 16 -pix_fmt yuv420p -r 25 -c:a aac -b:a 160k -movflags +faststart "$out" && echo "PANEL $out"
