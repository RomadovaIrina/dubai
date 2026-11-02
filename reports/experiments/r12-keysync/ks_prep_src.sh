#!/bin/bash
# r12 KS: crop + source-audio inputs for a video without pipeline artefacts: ks_prep_src.sh <id> <video>
set -e; id=$1; V=$2; K=/workspace/dub/keysync; cd /workspace/dub/dubai
/venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $V --out $K/lm/${id}_src_mp.npz | tail -1
/venv/dabai/bin/python - <<PY
import numpy as np; d = np.load("$K/lm/${id}_src_mp.npz"); lm = d["lm"][..., :2].astype(np.float32); W, H = [int(x) for x in d["size"]][::-1] if d["size"][0] > d["size"][1] else [int(x) for x in d["size"]]
if lm.max() < 2: lm = lm * np.array([W, H], np.float32)
np.savez("$K/lm/${id}_lm.npz", lm=lm); print("lm", lm.shape, lm.max(0).max(0), W, H)
PY
ffmpeg -v error -y -i $V -vn -ac 1 -ar 16000 $K/lm/${id}_src16k.wav; ffmpeg -v error -y -i $V -vn -c:a aac -b:a 160k $K/lm/${id}_src.m4a
cd $K && /venv/dabai/bin/python ks_prep.py $V lm/${id}_lm.npz lm/${id}_src16k.wav data $id
