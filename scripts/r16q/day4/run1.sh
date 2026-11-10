#!/bin/bash
# run1.sh ID SRC OURS -> /workspace/dub/q/out/<SET>/ID.json (+ series, det cache)
ID=$1; SRC=$2; OURS=$3; SET=${4:-now}; O=/workspace/dub/q/out/$SET; mkdir -p $O
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /workspace/dub/q/scripts/quality
/workspace/dub/qenv/bin/python r16q_defects.py --src "$SRC" --ours "$OURS" --out $O/$ID.json --npz $O/$ID.det.npz --label $ID > $O/$ID.log 2>&1; echo "$ID rc=$?" >> $O/done.txt
