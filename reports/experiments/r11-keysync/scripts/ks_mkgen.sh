#!/bin/bash
# KeySync layer dir in the r4m layout for r7_m7_comp.sh: ks_mkgen.sh <id> <gen tag> <KeySync out mp4> [ks_layer args]   (links the r9g_<id> master / timing / maskboxes)
# KS_SHIFT=auto (default): KeySync's AV offset differs per video (m18 -1, old03 +1 frame; constant within a video) -> the SyncNet offset of its own output vs the dub
set -e; id=$1; GT=$2; KS=$3; shift 3
B=/workspace/dub/artifacts/r4m; G=$B/r9g_$id; K=$B/$GT; W=$K/work_${id}_$GT; GW=$G/work_${id}_r9g_$id; mkdir -p $W
ln -sfn $G/timing $K/timing; ln -sfn $GW/video $W/video; cp $G/${id}_r9g_$id.manifest.json $K/${id}_$GT.manifest.json
sed "s#r9g_$id/work_${id}_r9g_$id/${id}_r9g_${id}_25fps#$GT/work_${id}_$GT/${id}_${GT}_25fps#g" $GW/${id}_r9g_${id}_25fps.manifest.json > $W/${id}_${GT}_25fps.manifest.json
ln -sfn $GW/${id}_r9g_${id}_25fps_maskboxes.npz $W/${id}_${GT}_25fps_maskboxes.npz; ln -sfn $GW/${id}_r9g_${id}_25fps.mp4 $W/${id}_${GT}_25fps.mp4
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
SH=${KS_SHIFT:-auto}
if [ "$SH" = auto ]; then
  SP=$K/ks_sync_raw.json; [ -s $SP ] || flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video $KS --audio /workspace/dub/artifacts/r4a/r9_${id}_htd/$id/dubbed_16k.wav --out $SP > /dev/null 2>&1
  SH=$(python3 -c "import json; print(-int(json.load(open('$SP'))['global']['av_offset']))"); echo "== KeySync AV offset -> shift $SH"
fi
python /workspace/dub/keysync/ks_layer.py --ks $KS --master $W/video/master_25fps.mp4 --box /workspace/dub/keysync/data/box_$id.json --out $W/${id}_${GT}_25fps_ls_raw.mp4 --shift $SH "$@"
flock /workspace/dub/gpu.lock /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $W/${id}_${GT}_25fps_ls_raw.mp4 --out $W/${id}_${GT}_25fps_ls_raw.mediapipe.npz | tail -1
