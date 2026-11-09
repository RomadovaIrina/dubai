#!/bin/bash
# r16 Q atlas clip: clip3.sh OUT.mp4 T0 DUR LABEL OLD.mp4 SRC.mp4 [NOW.mp4] -> 0.5x, panels "08.11 | now | source", height 480, audio = 08.11 at 0.5x
O=$1; T=$2; L=$3; LAB=$4; A=$5; S=$6; N=${7:-}
F=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf; dt() { [ -s $F ] && printf ",drawtext=fontfile=%s:text=%s:x=10:y=%s:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6" $F "$1" "${2:-10}"; }
in=(-ss $T -t $L -i $A); fc="[0:v]fps=25,scale=-2:480,setsar=1$(dt 08.11)[a]"; st="[a]"; k=1
if [ -n "$N" ]; then in+=(-ss $T -t $L -i $N); fc="$fc;[$k:v]fps=25,scale=-2:480,setsar=1$(dt now)[n]"; st="$st[n]"; k=$((k+1)); fi
in+=(-ss $T -t $L -i $S); fc="$fc;[$k:v]fps=25,scale=-2:480,setsar=1$(dt source)[s]"; st="$st[s]"; k=$((k+1))
fc="$fc;${st}hstack=inputs=$k,setpts=2*PTS$(dt "$LAB" h-36)[v];[0:a]atempo=0.5[au]"
nice ffmpeg -v error -threads 4 -y "${in[@]}" -filter_complex "$fc" -map "[v]" -map "[au]" -c:v libx264 -threads 4 -crf 27 -preset fast -c:a aac -b:a 64k -movflags +faststart $O && echo "CLIP $O $(stat -c %s $O)"
