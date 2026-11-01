#!/bin/bash
# r11 FACE: where does the composite lose SyncNet? usage: attr.sh <id> <gen tag> <layer mp4 (composite input)> <prefix> [variant names...]
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
id=$1; ST=$2; LS=$3; P=$4; shift 4
B=/workspace/dub/artifacts/r4m; W=$B/$ST/work_${id}_$ST; M=$W/${id}_${ST}_25fps.manifest.json; A=/workspace/dub/artifacts/r4a/r9_${id}_htd/$id/dubbed_16k.wav
[ -s $A ] || A=$(ls /workspace/dub/ksx/artifacts/r4a/r9_${id}_htd/$id/dubbed_16k.wav)
FB=$(ls $W/${id}_${ST}_25fps_fb_fb7_*rep2.npz 2>/dev/null | head -1); [ -n "$FB" ] || FB=$W/${id}_${ST}_25fps_fb_fb7.npz
V1="--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8 --sanitize-guard on --mouth-region 0.38 --paste-guard 0.5 --tex-tau 1000 --tex-sigma 2.5"
for v in "$@"; do
  case $v in
    base) X="--fallback $FB";; nofb) X="";; mr0) X="--fallback $FB --mouth-region 0";; tex0) X="--fallback $FB --skin-texture off";;
    san0) X="--fallback $FB --sanitize-layer 0";; reg0) X="--fallback $FB --register off";; near) X="--fallback $FB --retime nearest";;
    sh0) X="--fallback $FB --sharpen-auto 0 --sharpen-lips-max 0";; cm0) X="--fallback $FB --colour-match off";; lg0) X="--fallback $FB --lip-guard off";;
    fc0) X="--fallback $FB --face-clip off";; pg0) X="--fallback $FB --paste-guard 0";;
    bare) X="--mouth-region 0 --skin-texture off --sanitize-layer 0 --sharpen-auto 0 --sharpen-lips-max 0 --lip-guard off --colour-match off";;
    *) X="--fallback $FB $(eval echo \$X_$v)";;
  esac
  T=${P}_$v; FIX30_LS=$LS FIX30_MASTER_FPS=25 FIX30_TD=$B/$ST/timing FIX30_MODE=r3fix FIX30_THREADS=16 FIX30_ARGS="$V1 $X" bash scripts/quality/r3_fix30.sh $M $T > /tmp/attr_$T.log 2>&1
  O=$W/${id}_${ST}_25fps_$T.mp4; [ -s $O ] || { echo "== $v FAILED"; tail -3 $W/${id}_${ST}_25fps_$T.log; continue; }
  flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $O --audio $A --label "$id $v"
done
echo ATTR_DONE $P
