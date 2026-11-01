#!/bin/bash
# KeySync layer through the r9+fix composite (r7_m7_comp.sh, unchanged): ks_comp.sh <id> <TAG> <SRC_TAG>   (no CodeFormer-detail stage: KeySync has no pre-CF dump)
id=$1; TAG=$2; ST=$3; cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
flock /workspace/dub/gpu.lock env SKIP_SCORE=1 TAG=$TAG SRC_TAG=$ST SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$(cat reports/experiments/r8-face/fb7.args)" REP=${REP-1} EXTRA="--tex-tau 1000 --tex-sigma 2.5" bash scripts/r7/r7_m7_comp.sh $id
