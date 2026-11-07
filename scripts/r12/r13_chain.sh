#!/bin/bash
# r13 LS: the whole round-13 queue on a fresh box. setsid nohup bash scripts/r12/r13_chain.sh > /workspace/dub/r12/logs/chain.out 2>&1 &
#   setup -> r9 models -> pynvml -> per video (when its input is complete): k5 baseline + gate base -> warm control -> LS_RATE=2/3 (full gate)
#   -> LS_RATE=2/3 + ss10 (full gate). Order m03, old03, m18 (m03 first: the coordinator wants it first).
cd /workspace/dub/dubai; mkdir -p /workspace/dub/r12/logs; export E2E_METRICS=off   # production stages only (the gate scores separately)
until grep -q SETUP_DONE /workspace/dub/setup.log; do sleep 30; done; echo "== setup done $(date +%T)"
[ -f /workspace/dub/r9_models.done ] || { bash scripts/r9/r9_models.sh > /workspace/dub/r9_models.log 2>&1 && touch /workspace/dub/r9_models.done; }; tail -2 /workspace/dub/r9_models.log
/venv/main/bin/python -c "import pynvml" 2>/dev/null || uv pip install --python /venv/main/bin/python nvidia-ml-py 2>&1 | tail -1
M=/workspace/dub/inputs/mamakris; T=/workspace/dub/inputs/test_videos_raw
declare -A F=([m03]=$M/story-1774249919482-367803614.MOV [m18]=$M/story-1774250817106-616905094.mp4 [old03]=$T/03.mp4)
ready() {   # the upload is complete: size stable for 60 s and ffprobe reads a duration
  local f=$1 s0 s1; [ -f $f ] || return 1; s0=$(stat -c %s $f); sleep 60; s1=$(stat -c %s $f)
  [ "$s0" = "$s1" ] && ffprobe -v error -show_entries format=duration -of csv=p=0 $f > /dev/null 2>&1
}
P="--codeformer optimized --codeformer-w 1.0 --codeformer-parse off --codeformer-batch 16"
for id in m03 old03 m18; do
  until ready ${F[$id]}; do sleep 30; done; echo "== input $id ready $(date +%T)"
  NOFX=1 bash scripts/r12/ls_base.sh $id
  bash scripts/r12/ls_arm.sh $id t0c ""
  bash scripts/r12/ls_arm.sh $id r23 "" LS_RATE=2/3
  bash scripts/r12/ls_arm.sh $id r23s "$P --steps 10 --steps-silent 10" LS_RATE=2/3
  echo "== $id arms done $(date +%T)"
done
echo R13_CHAIN_DONE
