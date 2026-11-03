#!/bin/bash
# r12 LS: the k5 baselines (E2E_FX=1 E2E_SYNC=1, production e2e) of old03 / m03 / m18 on this box + the r9 + fix composite of the same layers
# (reference) + the gate bases. Layout as r10 (scripts/r10/gate.sh reads /workspace/dub/r10/runs/<id>): base = k5 (brief_r12: compare against k5).
#   setsid nohup bash scripts/r12/ls_base.sh old03 m03 m18 > /workspace/dub/r12/logs/base.out 2>&1 &
# No flock here: e2e.sh / r4_video.sh take gpu.lock around their own GPU stages (a nested flock deadlocks).
set -u
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
R=/workspace/dub/r10/runs; M=/workspace/dub/inputs/mamakris; T=/workspace/dub/inputs/test_videos_raw; B=/workspace/dub/artifacts/r4m; mkdir -p $R /workspace/dub/r12/logs
declare -A F=([m03]=$M/story-1774249919482-367803614.MOV [m18]=$M/story-1774250817106-616905094.mp4 [old03]=$T/03.mp4 [old05]=$T/05.mp4)
for id in "$@"; do
  [ -f $R/$id/${id}_DabAI_r9_demucs.mp4 ] || E2E_FX=1 E2E_SYNC=1 ID=$id bash scripts/r9/e2e.sh ${F[$id]} $R/$id > $R/$id.out 2>&1
  grep -E "R9_E2E_DONE|NVML GPU-busy" $R/$id.out $R/$id/logs/e2e.log 2>/dev/null | tail -2
  export DUB_INPUTS=$R/$id/work/in
  if [ "${NOFX:-0}" != 1 ]; then   # NOFX=1: no r9 + fix reference composite / its gate
  [ -f $B/r9f_${id}fx/${id}_r9f_${id}fx.mp4 ] || flock /workspace/dub/gpu.lock env SKIP_SCORE=1 TAG=r9f_${id}fx SRC_TAG=r9g_$id CFD=self CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4" \
    SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$(cat reports/experiments/r8-face/fb7.args)" REP=1 bash scripts/r7/r7_m7_comp.sh $id > /workspace/dub/r12/logs/${id}_fx.log 2>&1
  fi
  bash scripts/r10/gate.sh $id "" > /workspace/dub/r12/logs/gate_${id}base.log 2>&1
  if [ "${NOFX:-0}" != 1 ]; then
  bash scripts/r10/gate.sh $id FX $B/r9f_${id}fx/${id}_r9f_${id}fx.mp4 > /workspace/dub/r12/logs/gate_${id}FX.log 2>&1
  python scripts/r10/gate_cmp.py --base /workspace/dub/r10/gate/${id}FX --cand /workspace/dub/r10/gate/${id}base --md /workspace/dub/r12/gate.md | head -n 3
  fi
  echo "== $id base (k5) + r9fix reference + gates $(date +%T)"
done
echo R12_BASE_DONE "$@"
