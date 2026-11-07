#!/bin/bash
# r16 K1: 0.5x N-panel face clip with labels + time burnt in: k_clip3.sh <out.mp4> <t0> <t1> <box.json> <audio> <lab1>=<v1> <lab2>=<v2> ...
O=$1; T0=$2; T1=$3; B=$4; AU=$5; shift 5
read x y s < <(python3 -c "import json; b=json.load(open(\"$B\")); print(b[\"x\"], b[\"y\"], b[\"side\"])")
F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf; IN=(); FC=""; i=0; L=""
for p in "$@"; do lab=${p%%=*}; v=${p#*=}; IN+=(-ss $T0 -to $T1 -i $v)
  FC+="[$i:v]crop=$s:$s:$x:$y,scale=540:540:flags=lanczos,drawtext=fontfile=$F:text=${lab}:x=10:y=10:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.6,drawtext=fontfile=$F:text='%{pts\:hms\:$T0}':x=10:y=h-40:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6,setpts=2*(PTS-STARTPTS)[p$i];"; L+="[p$i]"; i=$((i + 1)); done
ffmpeg -nostdin -v error -y "${IN[@]}" -ss $T0 -to $T1 -i $AU -filter_complex "${FC}${L}hstack=inputs=$i[v];[$i:a]atempo=0.5[a]" -map "[v]" -map "[a]" -c:v libx264 -profile:v high -level 4.1 -crf 18 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart $O && echo CLIP $O
