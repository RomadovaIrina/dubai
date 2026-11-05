#!/bin/bash
# r15 K: 0.5x 3-panel A | B | C over windows, a fixed crop (env CROP=w:h:x:y, e.g. the KeySync box), each panel 400 px wide, audio from C.
# Labels via textfile (spaces / dashes safe).   CROP=... bash k_clip3.sh A "label A" B "label B" C "label C" OUT.mp4 "3-7 23-25"
A=$1; LA=$2; B=$3; LB=$4; C=$5; LC=$6; O=$7; WINS=$8
F=$(fc-list 2>/dev/null | grep -i "DejaVuSans-Bold.ttf" | head -1 | cut -d: -f1); FF=""; [ -n "$F" ] && FF="fontfile=$F:"
DT="x=6:y=6:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6"
T=$(mktemp -d); : > $T/l.txt; i=0
for w in $WINS; do
  s=${w%-*}; e=${w#*-}; d=$(python3 -c "print($e - $s)")
  printf '%s  %s s  0.5x' "$LA" "$w" > $T/la$i.txt; printf '%s' "$LB" > $T/lb$i.txt; printf '%s' "$LC" > $T/lc$i.txt
  V="crop=$CROP,scale=400:-2,setsar=1,fps=30,setpts=2*PTS"
  ffmpeg -loglevel error -y -ss $s -t $d -i $A -ss $s -t $d -i $B -ss $s -t $d -i $C -filter_complex \
    "[0:v]$V,drawtext=${FF}textfile=$T/la$i.txt:$DT[a];[1:v]$V,drawtext=${FF}textfile=$T/lb$i.txt:$DT[b];[2:v]$V,drawtext=${FF}textfile=$T/lc$i.txt:$DT[c];[a][b][c]hstack=inputs=3[v];[2:a]atempo=0.5,aresample=44100[au]" \
    -map "[v]" -map "[au]" -c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p -c:a aac -b:a 128k $T/w$i.mp4 && echo "file $T/w$i.mp4" >> $T/l.txt
  i=$((i + 1))
done
ffmpeg -loglevel error -y -f concat -safe 0 -i $T/l.txt -c copy -movflags +faststart $O; rm -rf $T; ls -la $O
