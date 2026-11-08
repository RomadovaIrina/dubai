#!/bin/bash
# r16 Q 08.11: score one e2e final for Irina's watch list: speech@0 (route, full length), all detectors vs the master, spring.
#   bash watch_one.sh ID FINAL JOBDIR     (JOBDIR = /workspace/dub/r16t/golden/jobs/<job>; dub = artifacts/r4a/r9_<job>/<job>/dubbed_16k.wav)
set -u
ID=$1; F=$(realpath "$2"); J=$(realpath "$3"); JN=$(basename $J); O=/workspace/dub/r16q/watch/$ID; mkdir -p $O
cd /workspace/dub/qa; PY=/venv/dabai/bin/python; MP=/venv/mediapipe/bin/python
M=$J/e2e/work/ks/master_25fps.mp4; RT=$J/e2e/work/route/route.json; RM=$J/e2e/work/route/master_25fps.mp4
D=/workspace/dub/artifacts/r4a/r9_$JN/$JN/dubbed_16k.wav
for x in $F $M $D; do [ -s $x ] || { echo "MISSING $x"; exit 2; }; done
cp $J/status.json $O/status.json 2>/dev/null
RA=(); if [ -s $RT ]; then if [ -s $RM ]; then WH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0:s=, $RM); else WH=$(python3 -c "import json;p=json.load(open('$J/status.json'))['input_probe'];print(p['width'],p['height'],sep=',')"); fi; RA=(--route $RT --route-wh $WH); fi
echo "WATCH $ID final=$F master=$M dub=$D route=${RT} wh=${WH:-}"
bash scripts/r16q/eval_final.sh $F $D $O/eval --noise 0 --device ${DEV:-cuda} "${RA[@]}" --label $ID > $O/eval.log 2>&1 &
nice $MP scripts/quality/r16q_detect.py --src $M --ours $F --out $O/det.json --npz $O/det.npz --label $ID > $O/det.log 2>&1 &
nice $PY scripts/quality/r16q_spring.py --final $F --master $M --out $O/spring.json > $O/spring.log 2>&1 &
wait; grep -h -E "^EVAL|^SPRING" $O/eval.log $O/spring.log | cut -c1-400; tail -1 $O/det.log | cut -c1-300; echo "WATCH_DONE $ID"
