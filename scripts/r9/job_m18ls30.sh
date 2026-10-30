#!/bin/bash
while kill -0 2203411 2>/dev/null; do sleep 20; done
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1 METRICS=off DUB_INPUTS=/workspace/dub/r9/runs/m18/work/in LS_FPS=30
A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; D=/workspace/dub/r9/diag; id=m18
t0=$(date +%s)
AUDIO=$A/r9_$id TAG=r9g_m18ls30 SKIP_SCORE=1 VIDEO_EXTRA="--codeformer optimized --codeformer-w 1.0 --steps 10 --steps-silent 20 --codeformer-parse off --codeformer-batch 16" bash scripts/r7/r7_e2e.sh $id > $D/m18ls30.gen.log 2>&1
t1=$(date +%s)
SKIP_SCORE=1 TAG=r9f_m18ls30fx SRC_TAG=r9g_m18ls30 CFD=self CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4" SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$(cat reports/experiments/r8-face/fb7.args)" REP=1 bash scripts/r7/r7_m7_comp.sh $id > $D/m18ls30.comp.log 2>&1
t2=$(date +%s); echo "== gen $((t1 - t0)) s comp $((t2 - t1)) s $(grep -h layer_repair $D/m18ls30.comp.log | cut -c1-200)"
python scripts/quality/r3_flash.py --video $B/r9f_m18ls30fx/m18_r9f_m18ls30fx.mp4 --src /workspace/dub/r9/runs/m18/work/in/m18.mp4 --lm $B/r9g_m18/timing/m18_source_lm_F.npz --out $D/flash_r9f_m18ls30fx.json > /dev/null 2>&1
echo M18LS30_DONE
