#!/bin/bash
# r16 L: 0.5x N-panel (512 each), label + source time burnt in, the dub audio at 0.5x.
#   l_panel.sh <out.mp4> <time offset s> <audio wav> label1=v1.mp4 label2=v2.mp4 ...
O=$1; OFF=$2; A=$3; shift 3; FT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
IN=""; FG=""; i=0; S=""
for p in "$@"; do l=${p%%=*}; v=${p#*=}; IN="$IN -i $v"
  FG="$FG[$i:v]scale=512:512,setpts=PTS-STARTPTS,drawtext=fontfile=$FT:text='$l':x=10:y=10:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.5[v$i];"; S="$S[v$i]"; i=$((i+1)); done
FG="$FG${S}hstack=inputs=$i,drawtext=fontfile=$FT:text='t %{eif\:t+$OFF\:d}.%{eif\:mod((t+$OFF)*100\,100)\:d\:2}':x=w/2-60:y=h-40:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.6,setpts=2*PTS[v];[$i:a]atempo=0.5[a]"
ffmpeg -v error -y $IN -i $A -filter_complex "$FG" -map "[v]" -map "[a]" -shortest -c:v libx264 -profile:v high -level 4.1 -crf 18 -pix_fmt yuv420p -movflags +faststart -c:a aac -b:a 128k $O
echo "panel $O"
