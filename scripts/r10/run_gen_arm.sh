#!/bin/bash
# r10 FACE generator arms: bash gen_arm.sh <id> <arm> "<VIDEO_EXTRA override>" [env...]  -> artifacts/r4m/r9g_<id><arm> (layers) + r9f_<id><arm> (r9 + fix composite)
set -u
id=$1; arm=$2; VX=$3; shift 3
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1 METRICS=off DUB_INPUTS=/workspace/dub/r10/runs/$id/work/in
A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; GT=r9g_$id$arm; FT=r9f_$id$arm; L=/workspace/dub/r10/arms; mkdir -p $L
t0=$(date +%s)
[ -f $B/$GT/work_${id}_$GT/${id}_${GT}_25fps.manifest.json ] || flock /workspace/dub/gpu.lock env "$@" AUDIO=$A/r9_$id TAG=$GT SKIP_SCORE=1 VIDEO_EXTRA="$VX" bash scripts/r7/r7_e2e.sh $id > $L/${id}$arm.gen.log 2>&1
t1=$(date +%s)
[ -f $B/$FT/${id}_$FT.mp4 ] || flock /workspace/dub/gpu.lock env SKIP_SCORE=1 TAG=$FT SRC_TAG=$GT CFD=self CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4" SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$(cat reports/experiments/r8-face/fb7.args)" REP=1 bash scripts/r7/r7_m7_comp.sh $id > $L/${id}$arm.comp.log 2>&1
echo "== arm $id$arm gen $((t1 - t0)) s comp $(( $(date +%s) - t1 )) s layer=$(ls $B/$GT/work_${id}_$GT/${id}_${GT}_25fps_ls_pre.mp4 2>/dev/null) final=$(ls $B/$FT/${id}_$FT.mp4 2>/dev/null)"
