#!/bin/bash
# r13 LS: a composite-only k5 variant on the cached layers of the k5 base (r9g_<id>), e.g. bash scripts/r12/comp_k5.sh m03 cms FC_CMS=1
#   -> artifacts/r4m/r9f_<id><tag>/<id>_r9f_<id><tag>.mp4   (CPU: CF-detail / SR / fallback / repair are cached under the k5 labels)
id=$1; tag=$2; shift 2
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1 DUB_INPUTS=/workspace/dub/r10/runs/$id/work/in
FBA="$(cat reports/experiments/r8-face/fb7.args) --open-gate 0.04 --dark-transient on --thr-wash-lips 0.2"
env "$@" SKIP_SCORE=1 TAG=r9f_$id$tag SRC_TAG=r9g_$id CFD=self CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4 --threads=12" SR=srt1 SR_EXTRA="--batch 8 --mask down2" \
  FB=fb7s FB_ARGS="$FBA" REP=1 REP_ARGS="--spikes off" REP_LABEL=rep2s FAST_ENC=ultrafast FC_EXTRA="--pre-threads=12 ${FCX:-}" \
  EXTRA="--tex-tau 1000 --tex-sigma 2.5 --mouth-region 0.55 --cm-mouth-excl 0.5" bash scripts/r7/r7_m7_comp.sh $id > /workspace/dub/r12/logs/comp_$id$tag.log 2>&1
ls -la /workspace/dub/artifacts/r4m/r9f_$id$tag/${id}_r9f_$id$tag.mp4 2>&1 | tail -1
