#!/bin/bash
# r11 FACE: KeySync soft lower-face paste variants -> SyncNet + face_metrics.  ksp.sh <id> <gen tag> <box> <layer suffix> <name> [extra args]
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; B=/workspace/dub/artifacts/r4m
id=$1; ST=$2; BOX=$3; LSX=$4; N=$5; shift 5
W=$B/$ST/work_${id}_$ST; P=$W/${id}_${ST}_25fps; SRC=/workspace/dub/r10/runs/$id/work/in/$id.mp4; SLM=$B/$ST/timing/${id}_source_lm_F.npz
A=/workspace/dub/artifacts/r4a/r9_${id}_htd/$id/dubbed_16k.wav; FIN=$(ls $B/r9f_$id/${id}_r9f_$id.mp4); O=/workspace/dub/r11/ksp; mkdir -p $O
python scripts/r11/ks_softpaste.py --layer ${P}_$LSX.mp4 --layer-lm ${P}_ls_srt1.mediapipe.npz --src $SRC --src-lm $SLM --box /workspace/dub/ksx/keysync/data/$BOX --audio $FIN --out $O/${id}_$N.mp4 "$@" | grep "^=="
s=$(flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $O/${id}_$N.mp4 --audio $A --label "$id $N")
python scripts/r10/face_metrics.py --video $O/${id}_$N.mp4 --src $SRC --lm-src $SLM --out $O/${id}_$N.json > /tmp/fm_ksp_${id}_$N.log 2>&1
f=$(python3 -c "
import json; j=json.load(open('$O/${id}_$N.json')); r=j.get('flicker',{}); c=j.get('chin_flow',{}) or {}
print('ring', round(r.get('ring_bright_ratio') or 0,2), 'ringS', r.get('ring_bright_strong'), '| chin swim', round(c.get('swim_rms') or 0,4), 'vel', round(c.get('vel_rms') or 0,4), 'ev', c.get('events'), 'S', c.get('strong'))" 2>&1 | tail -1)
echo "$s | $f"
