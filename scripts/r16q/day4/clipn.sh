#!/bin/bash
# clipn.sh OUT T0 DUR LABEL CROP(w:h:x:y in source px, or -) label=path ...   -> 0.5x, N panels side by side, each cropped to the face, height 560
O=$1; T=$2; L=$3; LAB=$4; CR=$5; shift 5; F=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf; n=0; in=(); fc=''; st=''
for p in "$@"; do lb=${p%%=*}; f=${p#*=}; in+=(-ss $T -t $L -i $f); c=""; [ "$CR" != - ] && c="crop=$CR,"
  fc="$fc[$n:v]fps=25,${c}scale=-2:560,setsar=1,drawtext=fontfile=$F:text=$lb:x=8:y=8:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6[v$n];"; st="$st[v$n]"; n=$((n+1)); done
nice ffmpeg -v error -threads 6 -y "${in[@]}" -filter_complex "${fc}${st}hstack=inputs=$n,setpts=2*PTS,drawtext=fontfile=$F:text=$LAB:x=8:y=h-34:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[v];[0:a]atempo=0.5[au]" -map "[v]" -map "[au]" -c:v libx264 -crf 24 -preset fast -c:a aac -b:a 64k $O && echo "$O $(stat -c %s $O)"
