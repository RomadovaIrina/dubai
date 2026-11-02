#!/bin/bash
# r12 KS lag fix on m18: tsmooth the fixed 25 fps layer (alpha 0.5 / 0.8), paste, lag + gate SyncNet
cd /workspace/dub/keysync; id=q18; O=final/$id; SRC=/workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4; A=/workspace/dub/artifacts/r4a/r9_${id}_htd/$id
B=$(python3 -c "import json; b=json.load(open('data/box_$id.json')); print(b['side'], b['x'], b['y'])"); set -- $B
for al in 0.5 0.8; do T=$O/ts$al; mkdir -p $T
  /venv/dabai/bin/python ks_tsmooth.py --layer $O/layer_fix.mp4 --lm $O/layerfix.mediapipe.npz --out $T/layer.mp4 --alpha $al
  ffmpeg -v error -y -i $T/layer.mp4 -vf "crop=$1:$1:$2:$3" -c:v libx264 -crf 10 $T/crop.mp4
  KS_SHIFT=0 /venv/dabai/bin/python ks_paste.py $T/crop.mp4 $SRC data/box_$id.json $A/final_mix_aac.m4a $T/paste.mp4
  ( cd /workspace/dub/dubai && /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video /workspace/dub/keysync/$T/paste.mp4 --out /workspace/dub/keysync/$T/paste.mediapipe.npz 2>/dev/null | tail -1 )
  /venv/dabai/bin/python ks_lag.py --lm $T/paste.mediapipe.npz --src-lm $O/src.mediapipe.npz --fps 30 --shift 1 --out $T/lag | cut -c1-120
  ( cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
    for v in /workspace/dub/keysync/$T/paste.mp4 /workspace/dub/outbox/r12_ks/final/m18_KeySync.mp4; do ffmpeg -v error -y -i $v -vf fps=25 -an -c:v libx264 -crf 12 /tmp/ts25.mp4; flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video /tmp/ts25.mp4 --audio $A/dubbed_16k.wav --out /tmp/ts.json > /dev/null 2>&1
      python3 -c "import json,statistics as st; d=json.load(open(\"/tmp/ts.json\")); sp=[p[\"conf_at0\"] for p in d[\"per_second\"] if p.get(\"conf_at0\") is not None and (p.get(\"dub_act\") or 0)>=0.5]; print(\"SYNC $v\", round(st.mean(sp),3))"; done )
done; echo LAGFIX_DONE
