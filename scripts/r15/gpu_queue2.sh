#!/bin/bash
# r15 F2/F3: ONE sequential GPU queue (replaces run_f2 step 5, gate_lc, gate_la, gate_la_m18, run_cf GPU parts).
# Before EVERY GPU step: no E2E_RUNNING marker (r15d / r15t), < 2 GB VRAM used. While a step runs, a watchdog kills it within 10 s if a
# marker appears (agent T's E2E needs ~18 GB); the step is then retried from scratch once the marker is gone. Never two of my GPU steps at once.
set -u; R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd $R; . scripts/env.sh >/dev/null; export DUB_ROOT=$R DUB_ALLOW_NC=1
B=/workspace/dub/artifacts/r4m; L=/workspace/dub/r15f3/logs; mkdir -p $L; MD=/workspace/dub/r15f2/gate_all.md
mk(){ [ -e /workspace/dub/r15d/E2E_RUNNING ] || [ -e /workspace/dub/r15t/E2E_RUNNING ]; }
clear_wait(){ while mk || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -ge 2048 ]; do sleep 20; done; }
kids(){ for c in $(ps -o pid= --ppid $1); do kids $c; echo $c; done; }
gstep(){   # gstep <name> <command...>: run under the rules, retry when pre-empted by a marker
  local name=$1; shift
  while true; do clear_wait; echo "== $name start $(date +%T)"
    if [ "${NOLOCK:-0}" = 1 ]; then "$@" > $L/$name.log 2>&1 & else flock /workspace/dub/gpu.lock "$@" > $L/$name.log 2>&1 & fi; local P=$!   # gates lock only their own GPU commands (gate_f.sh gw)
    while kill -0 $P 2>/dev/null; do if mk; then kill $(kids $P) $P 2>/dev/null; sleep 2; kill -9 $(kids $P) $P 2>/dev/null; echo "== $name PRE-EMPTED $(date +%T)"; break; fi; sleep 5; done
    wait $P 2>/dev/null; local rc=$?
    mk && continue
    echo "== $name done rc=$rc $(date +%T)"; return $rc
  done; }
gate(){ local id=$1 ST=$2 base=$3 sx=$4; local G=/workspace/dub/r15f/gate/$id$sx
  [ -s $G/face_gate.json ] && [ "${REGATE:-0}" = 0 ] || { rm -rf $G; NOLOCK=1 gstep gate_$id$sx env GT=$ST bash scripts/r15/gate_f.sh $id $sx; tail -n 1 $L/gate_$id$sx.log; }
  python scripts/r10/gate_cmp.py --base /workspace/dub/r15f/gate/$id$base --cand $G --md $MD | head -n 3; }
# r15 F3: which piece of the recipe costs m03 sync (8.60 -> 8.04) / pressed lips (+0.53 s)? (serialised with gpu_queue.sh by gpu.lock)
for sx in fub fubla fublac fublam fultc fublab; do gate q2_m03 r9g_q2_m03fu fu $sx; done
touch /workspace/dub/r15f3/GATES_DECOMP_DONE; echo QUEUE2_DONE $(date +%T)
