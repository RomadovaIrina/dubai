#!/bin/bash
# r16 T: 2-panel 0.5x eye clip with the time burnt in (source time, s): panel2.sh <src> <result> <out> [label]
SRC=$1; RES=$2; OUT=$3; LB=${4:-KeySync e2e}; F=$(fc-match -f "%{file}" "DejaVu Sans:bold"); T=$(mktemp)
cat > $T <<FG
[0:v]fps=25,scale=540:-2,drawtext=fontfile=$F:text=source:x=10:y=10:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5[a];
[1:v]fps=25,scale=540:-2,drawtext=fontfile=$F:text=$LB:x=10:y=10:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.5[b];
[a][b]hstack,drawtext=fontfile=$F:text='%{pts\\:hms}':x=(w-tw)/2:y=h-50:fontsize=34:fontcolor=yellow:box=1:boxcolor=black@0.6,setpts=2*PTS[v];
[1:a]atempo=0.5[au]
FG
ffmpeg -v error -y -i "$SRC" -i "$RES" -filter_complex_script $T -map "[v]" -map "[au]" -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -c:a aac -b:a 128k "$OUT"; rc=$?; rm -f $T; exit $rc
