#!/bin/bash
# r15 K: 0.5x 2-panel LEFT | RIGHT of the m03 mother / child boundary (fixed crop x 60 y 250 480x600 of 720x1280) over the worst cheek / chin moments
L=$1; LL=$2; R=$3; RL=$4; O=$5; WINS=${6:-"6.9-7.9 11.9-13.4 20.2-21.7 38.4-39.4"}
F=$(fc-list 2>/dev/null | grep -i "DejaVuSans-Bold.ttf" | head -1 | cut -d: -f1); FF=""; [ -n "$F" ] && FF="fontfile=$F:"; DT="x=8:y=8:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6"
T=$(mktemp -d); : > $T/l.txt; i=0
for w in $WINS; do s=${w%-*}; e=${w#*-}; d=$(python3 -c "print($e - $s)")
  ffmpeg -loglevel error -y -ss $s -t $d -i $L -ss $s -t $d -i $R -filter_complex "[0:v]crop=${CROP:-480:600:60:250},setsar=1,setpts=2*PTS,drawtext=${FF}text= s 0.5x:$DT[l];[1:v]crop=${CROP:-480:600:60:250},setsar=1,setpts=2*PTS,drawtext=${FF}text= s 0.5x:$DT[r];[l][r]hstack=inputs=2,fps=30[v];[0:a]atempo=0.5,aresample=44100[a]" -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p -c:a aac -b:a 128k $T/w$i.mp4 && echo "file $T/w$i.mp4" >> $T/l.txt; i=$((i + 1))
done
ffmpeg -loglevel error -y -f concat -safe 0 -i $T/l.txt -c copy -movflags +faststart $O; rm -rf $T; ls -la $O
