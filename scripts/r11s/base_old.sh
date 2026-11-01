#!/bin/bash
# r11 SYNC: the r9 + fix baselines (E2E_FX=1, production only) of old03 / old05 that r11-q lacked (as scripts/r10/base.sh, without the stab arm) + their gate base.
cd /workspace/dub/r11_sync; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/r11_sync DUB_ALLOW_NC=1
R=/workspace/dub/r10/runs; T=/workspace/dub/inputs/test_videos_raw
for id in "$@"; do
  f=$T/${id#old}.mp4
  [ -f $R/$id/${id}_DabAI_r9_demucs.mp4 ] || E2E_FX=1 E2E_METRICS=off ID=$id bash scripts/r9/e2e.sh $f $R/$id > $R/$id.out 2>&1
  grep -E "R9_E2E_DONE" $R/$id.out | tail -1
  bash /workspace/dub/dubai/scripts/r10/gate.sh $id "" > /workspace/dub/r11s/logs/gate_${id}base.log 2>&1; echo "== $id base + gate $(date +%T)"
done
