#!/bin/bash
# r16 T: 0.5x N-panel clip of the face box, the source time burnt in.
#   clip4.sh <out.mp4> <t0> <dur> <box.json> <audio> label=video [label=video ...]   (all videos on the same 25 fps grid / size; audio = the panel-2 one)
O=$1; T0=$2; DU=$3; BOX=$4; AU=$5; shift 5; F=$(fc-match -f "%{file}" "DejaVu Sans:bold")
read -r X Y S W H < <(python3 -c "import json; b=json.load(open('$BOX')); m=int(0.15*b['side']); x=max(0,b['x']-m); y=max(0,b['y']-m); s=min(b['side']+2*m, b['W']-x, b['H']-y); print(x, y, s, b['W'], b['H'])")
IN=(); FG=""; i=0
for a in "$@"; do L=${a%%=*}; V=${a#*=}; IN+=(-ss $T0 -t $DU -i $V)
  FG+="[$i:v]setpts=PTS-STARTPTS,fps=25,scale=$W:$H,crop=$S:$S:$X:$Y,scale=360:360,drawtext=fontfile=$F:text=$L:x=6:y=6:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.55[p$i];"; i=$((i + 1)); done
P=""; for j in $(seq 0 $((i - 1))); do P+="[p$j]"; done
FG+="${P}hstack=$i,drawtext=fontfile=$F:text='%{pts\:hms\:$T0}':x=(w-tw)/2:y=h-34:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.6,setpts=2*PTS[v];[$i:a]atempo=0.5[au]"
ffmpeg -v error -y "${IN[@]}" -ss $T0 -t $DU -i $AU -filter_complex "$FG" -map "[v]" -map "[au]" -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -movflags +faststart $O && echo "clip $O"
