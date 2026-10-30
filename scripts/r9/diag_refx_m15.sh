#!/bin/bash
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ALLOW_NC=1 METRICS=off DUB_ROOT=/workspace/dub/dubai DUB_INPUTS=/workspace/dub/r9/runs/m15/work/in
FBA="$(cat reports/experiments/r8-face/fb7.args)"; D=/workspace/dub/r9/diag
SKIP_SCORE=1 TAG=r9f_m15fx SRC_TAG=r9g_m15 CFD=self CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4" SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$FBA" REP=1 bash scripts/r7/r7_m7_comp.sh m15 > $D/fx_m15.log 2>&1
echo "== r9f_m15fx $(grep -h layer_repair $D/fx_m15.log | cut -c1-230) $(date +%T)"; echo REFX15_DONE
