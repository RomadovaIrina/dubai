#!/bin/bash
# r9 diag A/B 2: the m19b face with another LatentSync reference + an audio lead, on the SAME dub as the shipped r9 run.
#   bash ab2.sh <id>:<mode>:<lead> ...   -> artifacts/r4m/r9f_<id><mode>L<lead>/ (+ gate rows in the background: r4_gate score, frame gate, pressed lips, freeze)
set -u
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1 METRICS=off
A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; D=/workspace/dub/r9/diag
for spec in "$@"; do
  IFS=: read id m lead <<< "$spec"; s=$m"L"${lead/./}; G=r9g_$id$s; F=r9f_$id$s; export DUB_INPUTS=/workspace/dub/r9/runs/$id/work/in
  t0=$(date +%s); rm -rf $B/$G $B/$F
  AUDIO=$A/r9_$id TAG=$G SKIP_SCORE=1 VIDEO_EXTRA="--codeformer optimized --codeformer-w 1.0 --steps 10 --steps-silent 20 --codeformer-parse off --codeformer-batch 16 --ref-mode $m --audio-lead $lead" \
    bash scripts/r7/r7_e2e.sh $id > $D/ab2_${id}_$s.gen.log 2>&1
  t1=$(date +%s)
  SKIP_SCORE=1 TAG=$F SRC_TAG=$G CFD=self SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$(cat reports/experiments/r8-face/fb7.args)" bash scripts/r7/r7_m7_comp.sh $id > $D/ab2_${id}_$s.comp.log 2>&1
  t2=$(date +%s); echo "== $id $s gen $((t1 - t0)) s comp $((t2 - t1)) s $(date +%T)"
  [ -f $B/$F/${id}_$F.mp4 ] || { echo "FAIL $id $s"; continue; }
  ( export METRICS=on; python scripts/r4/r4_gate.py score --label $F --dir $B/$F --ids $id > $D/ab2_${id}_$s.score.log 2>&1
    bash scripts/r7/r7_frame_gate.sh $B/$F:$F $id > /dev/null 2>&1
    python scripts/r7/r7_closed_mouth.py --ids $id --variants $F:$A/r9_$id --out $D/ab2_${id}_$s.closed.json > /dev/null 2>&1
    python scripts/r7/r7_baseline.py freeze --tag $F --audio $A/r9_$id --ids $id --out $D/ab2_${id}_$s.gate.json > /dev/null 2>&1
    echo "SCORED $id $s" ) < /dev/null &
done
wait; echo AB2_DONE
