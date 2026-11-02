#!/bin/bash
# r12 KS finals: KeySync at the authors' 10-step quality, driven by the r9 dub voice; the layer shifted by its SyncNet AV offset;
# the face agent's ks_fix.py (r10-face scripts/r11) --guard 3 --chin tex; raw square paste back to the source frames; the demucs final mix.
#   ks_final.sh <id> <source video>  -> /workspace/dub/outbox/r12_ks/final/<name>_KeySync.mp4
set -e; id=$1; SRC=$2; NAME=${3:-$id}; K=/workspace/dub/keysync; A=/workspace/dub/artifacts/r4a/r9_${id}_htd/$id; G=/workspace/dub/artifacts/r4m/r9g_$id
W=$G/work_${id}_r9g_$id; MASTER=$W/video/master_25fps.mp4; SLM=$G/timing/${id}_source_lm_F.npz; O=$K/final/$id; mkdir -p $O /workspace/dub/outbox/r12_ks/final
cd $K
# 1) KeySync, 10 steps (the authors' settings), the dub voice
ffmpeg -v error -y -i $A/dubbed_16k.wav -ac 1 -ar 16000 data/audios/$id.wav
[ -s $O/ks/$id/${id}_$id.mp4 ] || { echo "$id $K/data/videos/$id.mp4 $K/data/audios/$id.wav" > $O/jobs.txt
  ( cd code && . $K/venv/bin/activate && flock /workspace/dub/gpu.lock python $K/ks_fast.py --jobs $O/jobs.txt --out $O/ks > $O/ks.log 2>&1 ); }
KS=$O/ks/$id/${id}_$id.mp4; grep -a KS_JOB $O/ks.log | cut -c1-140
cd /workspace/dub/dubai; . scripts/env.sh > /dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
# 2) AV offset -> shift; the layer in the master frames
[ -s $O/raw_sync.json ] || flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video $KS --audio $A/dubbed_16k.wav --out $O/raw_sync.json > /dev/null 2>&1
SH=$(python3 -c "import json; print(-int(json.load(open('$O/raw_sync.json'))['global']['av_offset']))"); echo "== $id AV offset -> shift $SH"
python $K/ks_layer.py --ks $KS --master $MASTER --box $K/data/box_$id.json --out $O/layer.mp4 --shift $SH --cm 0
/venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $O/layer.mp4 --out $O/layer.mediapipe.npz | tail -1
# 3) explosion guard + chin tex
python $K/r11face/ks_fix.py --layer $O/layer.mp4 --master $MASTER --box $K/data/box_$id.json --layer-lm $O/layer.mediapipe.npz --src-lm $SLM --out $O/layer_fix.mp4 --guard 3 --chin tex --guard-report $O/guard.json
# 4) raw paste at the source fps, the demucs mix
B=$(python3 -c "import json; b=json.load(open('$K/data/box_$id.json')); print(b['side'], b['x'], b['y'])"); set -- $B
ffmpeg -v error -y -i $O/layer_fix.mp4 -vf "crop=$1:$1:$2:$3" -c:v libx264 -crf 10 $O/crop_fix.mp4
KS_SHIFT=0 python $K/ks_paste.py $O/crop_fix.mp4 $SRC $K/data/box_$id.json $A/final_mix_aac.m4a $O/paste.mp4
# 5) the deliverable: <= 25 MB
D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $O/paste.mp4); BR=$(python3 -c "print(int(min(8000, 25e6 * 8 / $D * 0.92 / 1000 - 160)))")
ffmpeg -v error -y -i $O/paste.mp4 -c:v libx264 -preset slow -crf 18 -maxrate ${BR}k -bufsize $((2 * BR))k -pix_fmt yuv420p -profile:v high -movflags +faststart -c:a copy /workspace/dub/outbox/r12_ks/final/${NAME}_KeySync.mp4
ls -la /workspace/dub/outbox/r12_ks/final/${NAME}_KeySync.mp4; echo KS_FINAL_DONE $id
