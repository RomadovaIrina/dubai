#!/bin/bash
# LatentSync at the SOURCE fps (LS_FPS=30: no 25 -> 30 retime of the generated mouth) + the r9fx composite, on the shipped dub; then twitch / flash / gate
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1 METRICS=off LS_FPS=30
A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; D=/workspace/dub/r9/diag; FBA="$(cat reports/experiments/r8-face/fb7.args)"
for id in "$@"; do export DUB_INPUTS=/workspace/dub/r9/runs/$id/work/in; G=r9g_${id}ls30; T=r9f_${id}ls30fx; src=$(ls $DUB_INPUTS/$id.* | head -1)
  AUDIO=$A/r9_$id TAG=$G SKIP_SCORE=1 VIDEO_EXTRA="--codeformer optimized --codeformer-w 1.0 --steps 10 --steps-silent 20 --codeformer-parse off --codeformer-batch 16" bash scripts/r7/r7_e2e.sh $id > $D/ls30_$id.gen.log 2>&1
  flock /workspace/dub/gpu.lock env SKIP_SCORE=1 TAG=$T SRC_TAG=$G CFD=self CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4" SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$FBA" REP=1 bash scripts/r7/r7_m7_comp.sh $id > $D/ls30_$id.comp.log 2>&1
  echo "== $T $(grep -h layer_repair $D/ls30_$id.comp.log | cut -c1-160) $(date +%T)"
  flock /workspace/dub/gpu.lock python scripts/r4/r4_gate.py score --label $T --dir $B/$T --ids $id > $D/score_$T.log 2>&1
  python scripts/r7/r7_closed_mouth.py --ids $id --variants $T:$A/r9_$id --out $D/closed_$T.json > /dev/null 2>&1
  python scripts/quality/r3_flash.py --video $B/$T/${id}_$T.mp4 --src $src --lm $B/r9g_$id/timing/${id}_source_lm_F.npz --out $D/flash_$T.json > /dev/null 2>&1
  mkdir -p /workspace/dub/artifacts/r6f/twitch/$T; python scripts/r5/r5_twitch.py score --video $B/$T/${id}_$T.mp4 --src $src --lm /workspace/dub/artifacts/r6f/lm/${id}_src_lm.npz --manifest $B/$T/${id}_$T.manifest.json --out /workspace/dub/artifacts/r6f/twitch/$T/$id.twitch.json > /dev/null 2>&1
  echo "SCORED $T $(date +%T)"; done; echo LS30_DONE
