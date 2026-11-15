#!/bin/bash
# pair.sh <name> <seed> <t0> <t1> <work job (crop box)|-> <label=video> <label=video> [height]   -> out/pairs/<name>_BLIND.mp4 (+ .key.json): normal speed, sound of the first input
cd /root/r18; name=$1; seed=$2; t0=$3; t1=$4; wj=$5; a=$6; b=$7; h=${8:-960}
C=""; [ "$wj" != - ] && C="--crop auto --work /data/jobs/$wj/e2e/work"
mkdir -p out/pairs
bash code/scripts/r18/r18.sh sh "/venv/dabai/bin/python scripts/r18/r18_sbs.py --in $a --in $b --blind ${a%%=*},${b%%=*} --seed $seed --speed 1 --audio 0 --height $h --t0 $t0 --t1 $t1 $C --out /r18/out/pairs/${name}_BLIND.mp4" 2>&1 | grep R18_SBS | cut -c1-200
