#!/bin/bash
# r11 SYNC: a fallback / repair variant on the r9 + fix layers + its composite arm.
#   bash scripts/r11s/fb_arm.sh <id> <tag> "<extra r8_fallback flags (on top of fb7)>" "<r9_layer_repair flags>" "<extra composite flags>"
# REP=0: no layer repair (the new fallback straight into the composite).  -> /workspace/dub/r11s/fb/<id>/
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd $R; . scripts/env.sh >/dev/null; export DUB_ROOT=$R DUB_ALLOW_NC=1
id=$1; T=$2; FX=${3:-}; RX=${4:-}; CX=${5:-}; ST=r9g_$id; W=/workspace/dub/artifacts/r4m/$ST/work_${id}_$ST; P=$W/${id}_${ST}_25fps; O=/workspace/dub/r11s/fb/$id; mkdir -p $O
RAW=${P}_ls_cfd_ia04.mp4; SRN=${P}_ls_cfd_ia04_srt1.mp4; FBN=$O/fb_$T.npz
python scripts/r8/r8_fallback.py --work $W --ls $RAW --lm ${RAW%.mp4}.mediapipe.npz --out $FBN $(cat reports/experiments/r8-face/fb7.args) $FX | grep "^=="
if [ "${REP:-1}" = 1 ]; then
  LSR=$O/ls_rep_$T.mp4; FBR=$O/fb_rep_$T.npz
  python scripts/r9/r9_layer_repair.py --ls $SRN --lm ${SRN%.mp4}.mediapipe.npz --fb $FBN --src-lm $W/video/master_mediapipe.npz --src-video $W/video/master_25fps.mp4 --out $LSR --fb-out $FBR $RX | grep "^=="
  LS=$LSR FB=$FBR bash scripts/r11s/arm.sh $id $T "$CX"
else
  LS=$SRN FB=$FBN bash scripts/r11s/arm.sh $id $T "$CX"
fi
