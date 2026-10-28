#!/bin/bash
# Round 7 F: publish a merged version (r7_e2e.sh output, /workspace/dub/artifacts/r4m/<TAG>) as WATCH into /workspace/dub/outbox:
#   <id>_r7_<TAG>_WATCH.mp4, <id>_r7_<TAG>_WATCH_sbs.mp4 (PREV | TAG), r7_<TAG>_WATCH.md (gate rows vs PREV, whole-frame gate, twitch table,
#   whole-frame top-20 twitch sheets + half-speed clips in r7_<TAG>_twitch/, whole-frame gate sheets in r7_<TAG>_frame_gate/).
#   PREV=<dir>:<label> INTRO="..." bash scripts/r7/r7_watch.sh <TAG> [ids]      (default PREV = m5, the previous published version)
set -u
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
ARM=$1; shift; IDS=${*:-03 01 05 02 04}; A=/workspace/dub/artifacts/r4m/$ARM; OB=/workspace/dub/outbox; F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
PREV=${PREV:-/workspace/dub/artifacts/r4m/m5:m5}; PD=${PREV%%:*}; PL=${PREV##*:}; NAME=r7_$ARM; export OMP_NUM_THREADS=8
TW_PREV=${TW_PREV:-$PL}; mkdir -p $OB/${NAME}_frame_gate
for id in $IDS; do
  [ -f $A/${id}_$ARM.mp4 ] || { echo "no $ARM $id"; continue; }
  cp $A/${id}_$ARM.mp4 $OB/${id}_${NAME}_WATCH.mp4
  [ -f $PD/${id}_$PL.mp4 ] && ffmpeg -loglevel error -y -i $PD/${id}_$PL.mp4 -i $A/${id}_$ARM.mp4 -filter_complex "[0:v]drawtext=fontfile=$F:text='${PL^^}':x=12:y=12:fontsize=36:fontcolor=yellow:box=1:boxcolor=black@0.6[a];[1:v]drawtext=fontfile=$F:text='${ARM^^} (WATCH)':x=12:y=12:fontsize=36:fontcolor=yellow:box=1:boxcolor=black@0.6[b];[a][b]hstack=inputs=2,scale='min(2560,iw)':-2[v]" -map "[v]" -map 1:a:0 -c:v libx264 -crf 20 -preset veryfast -pix_fmt yuv420p -c:a copy $OB/${id}_${NAME}_WATCH_sbs.mp4
  T=/workspace/dub/artifacts/r6f/twitch/$ARM/${id}.twitch.json   # top-20 twitch sheet (WHOLE FRAME, r7) + 2-s half-speed clips
  [ -f $T ] && python scripts/r5/r5_twitch.py sheet --twitch $T --lm /workspace/dub/artifacts/r6f/lm/${id}_src_lm.npz --top 20 --outdir $OB/${NAME}_twitch --stem ${id}_$ARM --clips --whole-frame > /dev/null 2>&1
  G=/workspace/dub/artifacts/r7f/frame_gate/$ARM; [ -f $G/${id}_sheet.jpg ] && cp $G/${id}_sheet.jpg $OB/${NAME}_frame_gate/${id}_${ARM}_frame_gate_top12.jpg
  echo "== WATCH $ARM $id $(date +%T)"
done
IDC=$(echo $IDS | tr ' ' ',')
if [ -d $PD ] && ls $PD/*.score.log >/dev/null 2>&1; then
  python scripts/r4/r4_gate.py compare --new $A:$ARM --prev $PREV --ids $IDC --audio-info --out $A/gate_${ARM}_vs_$PL.md > $A/gate_compare.log 2>&1
  python3 scripts/r4/r4_watch_note.py --name $ARM --intro "${INTRO:-}" --new $A:$ARM --prev $PREV --gate $A/gate_${ARM}_vs_$PL.md --ids $IDS --out $OB/${NAME}_WATCH.md > /dev/null
  sed -i "s/^# r4 merged $ARM: WATCH candidate/# r7 $ARM: WATCH candidate/; s/<id>_r4_merged_${ARM}_WATCH/<id>_${NAME}_WATCH/g; s/(V4 | ${ARM^^})/(${PL^^} | ${ARM^^})/" $OB/${NAME}_WATCH.md
else
  { echo "# r7 $ARM: WATCH candidate"; echo; echo "${INTRO:-}"; echo; echo "Files: \`<id>_${NAME}_WATCH.mp4\`, \`<id>_${NAME}_WATCH_sbs.mp4\` (${PL^^} | ${ARM^^}). No score logs for $PL on this server: no r4_gate rows."; } > $OB/${NAME}_WATCH.md
fi
{ echo; echo "## Whole-frame gate (r7_frame_gate.py: % of the frame changed outside the mouth region, per frame; PASS = no frame > 0.2 %; sheets in \`${NAME}_frame_gate/\`)"
  echo "| video | verdict | worst frame | worst % | frames > 0.2 % | p99 % | face frames |"; echo "|---|---|---|---|---|---|---|"
  for id in $IDS; do python3 -c "
import json,sys; d=json.load(open('/workspace/dub/artifacts/r7f/frame_gate/$ARM/$id.json')); w=d['worst'][0] if d['worst'] else {}
print(f\"| $id | {d['verdict']} | {w.get('frame')} ({w.get('t_s')} s) | {d['worst_pct']} | {d['frames_over']} | {d['p99_pct']} | {d['frames_with_face']}/{d['frames']} |\")" 2>/dev/null; done
  echo; echo "## Face twitch detector (r5_twitch.py; lower = better; acceptance: NEW <= PREV on all 5 + the top-20 WHOLE-FRAME human review in \`${NAME}_twitch/\`)"
  python scripts/r6/r6_twitch_table.py v4 $TW_PREV $ARM --top 5 2>/dev/null
  [ -f $A/twitch_note.md ] && cat $A/twitch_note.md; } >> $OB/${NAME}_WATCH.md
echo R7_WATCH_DONE $ARM
