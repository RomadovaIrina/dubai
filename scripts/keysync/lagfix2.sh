#!/bin/bash
# r12 KS lag fix on m18 (redo, clean): tsmooth alpha 0.8, paste; gate SyncNet + lag for the final, ts0.5, ts0.8
cd /workspace/dub/keysync; id=q18; O=final/$id; SRC=/workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4; A=/workspace/dub/artifacts/r4a/r9_${id}_htd/$id
read S X Y <<< "$(python3 -c "import json; b=json.load(open('data/box_$id.json')); print(b['side'], b['x'], b['y'])")"
T=$O/ts0.8; mkdir -p $T
/venv/dabai/bin/python ks_tsmooth.py --layer $O/layer_fix.mp4 --lm $O/layerfix.mediapipe.npz --out $T/layer.mp4 --alpha 0.8
ffmpeg -v error -y -i $T/layer.mp4 -vf "crop=$S:$S:$X:$Y" -c:v libx264 -crf 10 $T/crop.mp4
KS_SHIFT=0 /venv/dabai/bin/python ks_paste.py $T/crop.mp4 $SRC data/box_$id.json $A/final_mix_aac.m4a $T/paste.mp4
( cd /workspace/dub/dubai && /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video /workspace/dub/keysync/$T/paste.mp4 --out /workspace/dub/keysync/$T/paste.mediapipe.npz 2>/dev/null | tail -1 )
/venv/dabai/bin/python ks_lag.py --lm $T/paste.mediapipe.npz --src-lm $O/src.mediapipe.npz --fps 30 --shift 1 --out $T/lag | cut -c1-110
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
for v in /workspace/dub/outbox/r12_ks/final/m18_KeySync.mp4 /workspace/dub/keysync/$O/ts0.5/paste.mp4 /workspace/dub/keysync/$T/paste.mp4; do
  n=$(echo $v | md5sum | cut -c1-6); ffmpeg -v error -y -i $v -vf fps=25 -an -c:v libx264 -crf 12 /tmp/s25_$n.mp4
  flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video /tmp/s25_$n.mp4 --audio $A/dubbed_16k.wav --out /tmp/s_$n.json > /tmp/s_$n.log 2>&1
  python3 -c "import json,statistics as st; d=json.load(open('/tmp/s_$n.json')); sp=[p['conf_at0'] for p in d['per_second'] if p.get('conf_at0') is not None and (p.get('dub_act') or 0)>=0.5]; print('SYNC', '$v', d['global'], round(st.mean(sp),3))"
done; echo LAGFIX2_DONE
