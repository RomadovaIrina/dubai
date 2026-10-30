#!/bin/bash
# fx-rep = layer repair without spike smoothing, FULL CodeFormer detail (no inner-mouth weight): isolates the detail-weight effect
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1 METRICS=off
A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; D=/workspace/dub/r9/diag; FBA="$(cat reports/experiments/r8-face/fb7.args)"
while pgrep -f "bash /workspace/dub/r9/diag/lite.sh" > /dev/null; do sleep 20; done
for id in "$@"; do export DUB_INPUTS=/workspace/dub/r9/runs/$id/work/in; T=r9f_${id}fxrep; src=$(ls $DUB_INPUTS/$id.* | head -1)
  flock /workspace/dub/gpu.lock env SKIP_SCORE=1 TAG=$T SRC_TAG=r9g_$id CFD=self SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$FBA" REP=1 REP_LABEL=reponly REP_ARGS="--spikes off" bash scripts/r7/r7_m7_comp.sh $id > $D/rep_$id.log 2>&1
  flock /workspace/dub/gpu.lock python scripts/r4/r4_gate.py score --label $T --dir $B/$T --ids $id > $D/score_$T.log 2>&1
  python scripts/r7/r7_closed_mouth.py --ids $id --variants $T:$A/r9_$id --out $D/closed_$T.json > /dev/null 2>&1
  python scripts/quality/r3_flash.py --video $B/$T/${id}_$T.mp4 --src $src --lm $B/r9g_$id/timing/${id}_source_lm_F.npz --out $D/flash_$T.json > /dev/null 2>&1
  mkdir -p /workspace/dub/artifacts/r6f/twitch/$T; python scripts/r5/r5_twitch.py score --video $B/$T/${id}_$T.mp4 --src $src --lm /workspace/dub/artifacts/r6f/lm/${id}_src_lm.npz --manifest $B/$T/${id}_$T.manifest.json --out /workspace/dub/artifacts/r6f/twitch/$T/$id.twitch.json > /dev/null 2>&1
  echo "SCORED $T $(date +%T)"; done; echo REPONLY_DONE
