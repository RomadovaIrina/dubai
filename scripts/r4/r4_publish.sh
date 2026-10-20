#!/bin/bash
# Round 4 publish (gate first): bash scripts/r4/r4_publish.sh <kind video|merged> <N> <new_dir:label> <prev_dir:label> [ids]
#  1. r4_gate.py compare NEW vs PREV (all five) -> outbox/r4_<kind>_vN_regression.md; any FAIL -> nothing is published (exit 1)
#  2. per id: outbox/<id>_r4_<kind>_vN.mp4, side by side V4 (reference, or SBS_REF) | NEW (NEW's audio) <id>_r4_<kind>_vN_sbs.mp4, review sheet
#     <id>_r4_<kind>_vN_review.md (one row per 10 s, NEW vs PREV on the same frame grid)
#  SBS_REF=dir:label:caption (env) = the left half of the side by side instead of the v4 reference finals, e.g. the previous published version
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
KIND=$1; N=$2; NEW=$3; PREV=$4; ND=${3%%:*}; NL=${3#*:}; PD=${4%%:*}; PL=${4#*:}; shift 4; IDS=${*:-01 02 03 04 05}
OB=/workspace/dub/outbox; SBS=${SBS_REF:-/workspace/dub/reference/v4:v4:V4}; RD=${SBS%%:*}; RL=${SBS#*:}; RC=${RL#*:}; RL=${RL%%:*}; LM=/workspace/dub/artifacts/r3d/timing; F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
V=${VNAME:-r4_${KIND}_v$N}; C=$(git rev-parse --short HEAD); REG=$OB/${V}_regression.md
python scripts/r4/r4_gate.py compare --new $NEW --prev $PREV --ids $(echo $IDS | tr ' ' ',') ${GATE_ARGS:-} --out $REG > /dev/null 2>&1; rc=$?
sed -i "1a\\
\\
Version **$V**, commit \`$C\`, NEW = \`$ND/*_$NL.mp4\`, PREV (gate baseline) = \`$PD/*_$PL.mp4\`, $(date -u +%FT%TZ)." $REG
[ $rc -eq 0 ] || { echo "GATE FAIL: $REG (nothing published)"; grep "^- " $REG; exit 1; }
lab() { echo "drawtext=fontfile=$F:text='$1':x=12:y=12:fontsize=36:fontcolor=yellow:box=1:boxcolor=black@0.6"; }
for id in $IDS; do
  cp $ND/${id}_$NL.mp4 $OB/${id}_$V.mp4
  ffmpeg -loglevel error -y -i $RD/${id}_$RL.mp4 -i $ND/${id}_$NL.mp4 -filter_complex "[0:v]$(lab "$RC")[a];[1:v]$(lab "NEW $V")[b];[a][b]hstack=inputs=2,scale='min(2560,iw)':-2[v]" \
    -map "[v]" -map 1:a:0 -c:v libx264 -crf 20 -preset veryfast -pix_fmt yuv420p -c:a copy $OB/${id}_${V}_sbs.mp4
  python scripts/r4/r4_review.py --id $id --new $ND/${id}_$NL.mp4 --prev $PD/${id}_$PL.mp4 --new-score $ND/${id}_$NL.score.json --prev-score $PD/${id}_$PL.score.json \
    --lm $LM/${id}_source_lm_F.npz --title "$V ($C) vs $PL" --out $OB/${id}_${V}_review.md > /dev/null
  echo "== $id -> $OB/${id}_$V.mp4 (+ _sbs, _review.md)"
done
echo "PUBLISHED $V ($C): $REG"
