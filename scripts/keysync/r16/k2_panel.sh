#!/bin/bash
# r16 K2: k2_panel.sh OUT.mp4 T0 T1 CROP(w:h:x:y in the master) AUDIO label=video ...  -> side-by-side face crops, time burnt in, 0.5x (video + audio)
out=$1; t0=$2; t1=$3; cr=$4; au=$5; shift 5; F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
ins=""; fc=""; i=0; st=""
for lv in "$@"; do l=${lv%%=*}; v=${lv#*=}; ins="$ins -ss $t0 -to $t1 -i $v"
  fc="$fc[$i:v]crop=$cr,scale=-2:540:flags=lanczos,setpts=PTS-STARTPTS,drawtext=fontfile=$F:text='$l':x=8:y=8:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.5,drawtext=fontfile=$F:text='%{pts\:hms\:$t0}':x=8:y=h-36:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.5,setpts=2*PTS[v$i];"
  st="$st[v$i]"; i=$((i+1)); done
ffmpeg -v error -y $ins -ss $t0 -to $t1 -i $au -filter_complex "${fc}${st}hstack=inputs=$i[v];[$i:a]asetpts=PTS-STARTPTS,atempo=0.5[a]" -map "[v]" -map "[a]" -c:v libx264 -crf 16 -preset medium -pix_fmt yuv420p -r 25 -c:a aac -b:a 160k $out && echo "PANEL $out"
