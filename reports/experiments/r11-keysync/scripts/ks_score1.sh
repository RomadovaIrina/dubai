#!/bin/bash
# score one more arm: ks_score1.sh <id> <src> <label>
id=$1; SRC=$2; T=$3; cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1 DUB_INPUTS=/workspace/dub/keysync/in_$id
B=/workspace/dub/artifacts/r4m; M=/workspace/dub/keysync/metrics/$id
[ -f $B/$T/${id}_$T.manifest.json ] || cp $B/r9f_$id/${id}_r9f_$id.manifest.json $B/$T/${id}_$T.manifest.json
python scripts/r4/r4_gate.py score --label $T --dir $B/$T --ids $id > $M/score_$T.log 2>&1; tail -1 $M/score_$T.log
python scripts/r10/face_metrics.py --video $B/$T/${id}_$T.mp4 --src $SRC --lm-src $B/r9g_$id/timing/${id}_source_lm_F.npz --out $M/F_$T.json --per-frame > $M/fm_$T.log 2>&1
python3 scripts/r10/tab.py $M; echo KS_SCORE1_DONE $T
