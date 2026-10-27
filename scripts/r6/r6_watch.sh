#!/bin/bash
# Round 6 F: publish an ARM (r6_arm.sh output) as WATCH: <id>_r6_<arm>_WATCH.mp4 + _sbs.mp4 (PREV | NEW) + r6_<arm>_WATCH.md (gate rows vs PREV,
# timecodes, twitch table, top-20 twitch sheets + half-speed clips) into /workspace/dub/outbox.
#   PREV=<dir>:<label> INTRO="..." bash scripts/r6/r6_watch.sh <arm> [ids]      (default PREV = m3, the previous published version)
set -u
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
ARM=$1; shift; IDS=${*:-05 04 03 01 02}; A=/workspace/dub/artifacts/r6f/arms/$ARM; OB=/workspace/dub/outbox; F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
PREV=${PREV:-/workspace/dub/artifacts/r4m/m3:m3}; PD=${PREV%%:*}; PL=${PREV##*:}; NAME=r6_$ARM; export OMP_NUM_THREADS=8
for id in $IDS; do
  [ -f $A/${id}_$ARM.mp4 ] || { echo "no $ARM $id"; continue; }
  cp $A/${id}_$ARM.mp4 $OB/${id}_${NAME}_WATCH.mp4
  ffmpeg -loglevel error -y -i $PD/${id}_$PL.mp4 -i $A/${id}_$ARM.mp4 -filter_complex "[0:v]drawtext=fontfile=$F:text='${PL^^}':x=12:y=12:fontsize=36:fontcolor=yellow:box=1:boxcolor=black@0.6[a];[1:v]drawtext=fontfile=$F:text='${ARM^^} (WATCH)':x=12:y=12:fontsize=36:fontcolor=yellow:box=1:boxcolor=black@0.6[b];[a][b]hstack=inputs=2,scale='min(2560,iw)':-2[v]" -map "[v]" -map 1:a:0 -c:v libx264 -crf 20 -preset veryfast -pix_fmt yuv420p -c:a copy $OB/${id}_${NAME}_WATCH_sbs.mp4
  # top-20 twitch sheet + 2-s half-speed clips of the NEW version (the human review of the acceptance rule)
  T=/workspace/dub/artifacts/r6f/twitch/$ARM/${id}.twitch.json
  [ -f $T ] && python scripts/r5/r5_twitch.py sheet --twitch $T --lm /workspace/dub/artifacts/r6f/lm/${id}_src_lm.npz --top 20 --outdir $OB/${NAME}_twitch --stem ${id}_$ARM --clips > /dev/null 2>&1
  echo "== WATCH $ARM $id $(date +%T)"
done
IDC=$(echo $IDS | tr ' ' ',')
python scripts/r4/r4_gate.py compare --new $A:$ARM --prev $PREV --ids $IDC --audio-info --out $A/gate_${ARM}_vs_$PL.md > $A/gate_compare.log 2>&1
python3 scripts/r4/r4_watch_note.py --name $ARM --intro "${INTRO:-}" --new $A:$ARM --prev $PREV --gate $A/gate_${ARM}_vs_$PL.md --ids $IDS --out $OB/${NAME}_WATCH.md > /dev/null
{ echo; echo "## Face twitch detector (r5_twitch.py; lower = better; acceptance: NEW <= v4 on all 5 + the top-20 human review in \`${NAME}_twitch/\`)"
  python scripts/r6/r6_twitch_table.py v4 $PL $ARM --top 5 2>/dev/null | sed "s/| m3loc |/| m3 |/"; } >> $OB/${NAME}_WATCH.md
echo R6_WATCH_DONE $ARM
