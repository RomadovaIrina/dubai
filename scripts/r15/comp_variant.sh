#!/bin/bash
# r15 W0: re-run ONLY the face_comp stage (scripts/r7/r7_m7_comp.sh, as scripts/r15/arm.sh) of a finished arm's generator layer with other
# composite flags -> a new arm. Used for TBDub: our fps_composite costs its layer -1.3 SyncNet (old03: SR layer 7.55 -> final 6.27; LatentSync
# loses 0.5), and --mouth-region (the paste capped to the lips + jaw drop, chin / jaw = source: right for LatentSync, which never moves the jaw)
# is the prime suspect for a generator that regenerates the whole lower face.
#   SRC=<arm> V=<new arm> MR=<mouth region, 0 = off> ARM_CFD=off|self bash scripts/r15/comp_variant.sh <id> ...
set -u
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
export DUB_ALLOW_NC=1 DUB_ALLOW_NC_BED=1 METRICS=off GEN_COMP=0 FAST_FB_BG=1 FAST_ENC=ultrafast CFD_EXTRA=--threads=12 FC_CMS=1
SRC=${SRC:?SRC=<arm with the generator layer>}; V=${V:?V=<new arm>}; MR=${MR:-0}; X=/workspace/dub/r15w/eval; A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m
export FC_EXTRA="--pre-threads=12 --sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2 --reg-smooth 4 --mouth-region $MR"
for id in "$@"; do
  ID=q2_$id; GT=r9g_${ID}_$SRC; FT=r9f_${ID}_$V; O=$X/${V}_$id; L=$O/logs; mkdir -p $L; rm -rf $B/$FT
  export DUB_INPUTS=$X/$ID/e2e/work/in
  CFDV=self; [ "${ARM_CFD:-self}" = off ] && CFDV=""
  FBA="$(cat reports/experiments/r8-face/fb7.args) --open-gate 0.04 --dark-transient on --thr-wash-lips 0.2"
  REP=1 CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4 $CFD_EXTRA" REP_ARGS="--spikes off" REP_LABEL=rep2s \
  SKIP_SCORE=1 TAG=$FT SRC_TAG=$GT CFD=$CFDV SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7s FB_ARGS="$FBA" EXTRA="--tex-tau 1000 --tex-sigma 2.5 --mouth-region $MR --cm-mouth-excl 0.5" \
    bash scripts/r7/r7_m7_comp.sh $ID > $L/face_comp.log 2>&1
  VID=$B/$FT/${ID}_$FT.mp4
  [ -f $VID ] || { echo "== ${V}_$id FAIL (see $L/face_comp.log)"; continue; }
  ffmpeg -loglevel error -y -i $VID -i $A/r9_${ID}_htd/$ID/final_mix_aac.m4a -map 0:v:0 -map 1:a:0 -c copy -movflags +faststart $O/${V}_$id.mp4
  flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $O/${V}_$id.mp4 --audio $A/r9_$ID/$ID/dubbed_16k.wav --label ${V}_$id 2>&1 | grep "^== sync" | tee $X/${V}_$id.sync.txt
done
