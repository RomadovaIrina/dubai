#!/bin/bash
# KeySync bake-off scoring: r4_gate score (SyncNet on dub speech, strong flashes, ...) for KeySync and the r9+fix reference + face_metrics (stick / slide / swim)
# ks_score.sh <id> <src video> ; the KeySync arm is artifacts/r4m/ks_<id>/<id>_ks_<id>.mp4
id=$1; SRC=$2; cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
B=/workspace/dub/artifacts/r4m; M=/workspace/dub/keysync/metrics/$id; mkdir -p $M /workspace/dub/keysync/in_$id
ln -sfn $SRC /workspace/dub/keysync/in_$id/$id.${SRC##*.}; export DUB_INPUTS=/workspace/dub/keysync/in_$id
cp -n $B/r9f_$id/${id}_r9f_$id.manifest.json $B/ks_$id/${id}_ks_$id.manifest.json
for T in r9f_$id ks_$id; do python scripts/r4/r4_gate.py score --label $T --dir $B/$T --ids $id > $M/score_$T.log 2>&1; tail -1 $M/score_$T.log; done
SLM=$B/r9g_$id/timing/${id}_source_lm_F.npz
for T in r9f_$id ks_$id; do python scripts/r10/face_metrics.py --video $B/$T/${id}_$T.mp4 --src $SRC --lm-src $SLM --out $M/F_$T.json --per-frame > $M/fm_$T.log 2>&1; done
python3 scripts/r10/tab.py $M; echo KS_SCORE_DONE $id
