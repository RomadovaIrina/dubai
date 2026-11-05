#!/bin/bash
# r15 F: composite-only rerun of an r14 q2 final (E2E_FX=1 E2E_SYNC=1 + q2 + F3, as in scripts/r9/e2e.sh) on its cached layers, with its OWN
# layer-repair label (never overwrites the production rep2s layer) -> artifacts/r4m/r9f_<run><sfx>/<run>_r9f_<run><sfx>.mp4
#   bash scripts/r15/comp_f.sh <run, e.g. q2_m03> <sfx> <repair label> "<extra r9_layer_repair flags>"
set -u; run=$1; sfx=$2; RL=$3; RX=${4:-}
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd $R; . scripts/env.sh >/dev/null; export DUB_ROOT=$R DUB_ALLOW_NC=1
Q2="--sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2"; F3=${F3:-"--reg-smooth 4 --mouth-region 0.45"}   # F3=... overrides the composite base
FBA="$(cat reports/experiments/r8-face/fb7.args) --open-gate 0.04 --dark-transient on --thr-wash-lips 0.2"
mkdir -p /workspace/dub/r15f/logs
FC_CMS=1 LS_CMS_READ=1 SKIP_SCORE=1 TAG=r9f_$run$sfx SRC_TAG=${GT:-r9g_$run} CFD=self CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4 --threads=12" SR=srt1 SR_EXTRA="--batch 8 --mask down2" \
  FB=fb7s FB_ARGS="$FBA" REP=1 REP_ARGS="--spikes off $RX" REP_LABEL=$RL FC_EXTRA="--pre-threads=12 $Q2 $F3" \
  EXTRA="--tex-tau 1000 --tex-sigma 2.5 --mouth-region 0.55 --cm-mouth-excl 0.5" bash scripts/r7/r7_m7_comp.sh $run > /workspace/dub/r15f/logs/comp_$run$sfx.log 2>&1
ls -la /workspace/dub/artifacts/r4m/r9f_$run$sfx/${run}_r9f_$run$sfx.mp4 2>&1 | tail -1; grep "== layer_repair" /workspace/dub/r15f/logs/comp_$run$sfx.log | cut -c1-400
