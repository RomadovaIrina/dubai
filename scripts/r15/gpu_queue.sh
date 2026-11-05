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
    flock /workspace/dub/gpu.lock "$@" > $L/$name.log 2>&1 & local P=$!
    while kill -0 $P 2>/dev/null; do if mk; then kill $(kids $P) $P 2>/dev/null; sleep 2; kill -9 $(kids $P) $P 2>/dev/null; echo "== $name PRE-EMPTED $(date +%T)"; break; fi; sleep 5; done
    wait $P 2>/dev/null; local rc=$?
    mk && continue
    echo "== $name done rc=$rc $(date +%T)"; return $rc
  done; }
gate(){ local id=$1 ST=$2 base=$3 sx=$4; local G=/workspace/dub/r15f/gate/$id$sx
  [ -s $G/face_gate.json ] && [ "${REGATE:-0}" = 0 ] || { rm -rf $G; gstep gate_$id$sx env GT=$ST bash scripts/r15/gate_f.sh $id $sx; tail -n 1 $L/gate_$id$sx.log; }
  python scripts/r10/gate_cmp.py --base /workspace/dub/r15f/gate/$id$base --cand $G --md $MD | head -n 3; }
for i in $(seq 1 45); do mk && break; sleep 20; done   # agent T is restarting its run: give its marker up to 15 min to appear first
# 0. base gates (production a), only if missing
for x in q2_old03:r9g_q2_old03:c0 q2_m18:r9g_q2_m18:c0 q2_m03:r9g_q2_m03fu:fu; do IFS=: read id ST A <<< "$x"; G=/workspace/dub/r15f/gate/$id$A
  [ -s $G/face_gate.json ] || { rm -rf $G; gstep gate_$id$A env GT=$ST bash scripts/r15/gate_f.sh $id $A; }; done
# 1. the clean candidate first: b + adaptive lipcon on all three
until [ -s $B/r9f_q2_m18bla/q2_m18_r9f_q2_m18bla.mp4 ]; do sleep 30; done
gate q2_m03 r9g_q2_m03fu fu fubla; gate q2_old03 r9g_q2_old03 c0 bla; gate q2_m18 r9g_q2_m18 c0 bla
touch /workspace/dub/r15f2/GATES_CLEAN_DONE
# 2. the rest of F2 (b / c, fixed lipcon, c + adaptive)
gate q2_m03 r9g_q2_m03fu fu fub; gate q2_m03 r9g_q2_m03fu fu fuc; gate q2_m18 r9g_q2_m18 c0 b; gate q2_m18 r9g_q2_m18 c0 c
gate q2_m03 r9g_q2_m03fu fu fucla; gate q2_old03 r9g_q2_old03 c0 cla
gate q2_m03 r9g_q2_m03fu fu ful; gate q2_m03 r9g_q2_m03fu fu fubl; gate q2_m03 r9g_q2_m03fu fu fucl
touch /workspace/dub/r15f2/GATES_F2_DONE
# 3. F3 CodeFormer sweep: CF layers + SR (GPU), the detail layers on the CPU in between
for x in q2_m03:r9g_q2_m03fu q2_old03:r9g_q2_old03; do IFS=: read id ST <<< "$x"; W=$B/$ST/work_${id}_$ST; P=$W/${id}_${ST}_25fps; LMN=${P}_ls_raw.mediapipe.npz
  for w in 05 07 10; do CF=${P}_ls_cfw$w.mp4; D=${P}_ls_cfd_w$w.mp4
    [ -s $D ] || { [ -s $CF ] || gstep cf_$id$w python scripts/r15/cf_layer.py --work $W --pre ${P}_ls_pre.mp4 --lm $LMN --w 0.${w#0} --out $CF
      python scripts/r8/r8_cf_detail.py --base ${P}_ls_pre.mp4 --cf $CF --lm $LMN --out $D --inner-alpha 0.4 | grep "^=="; }
    cp $LMN ${D%.mp4}.mediapipe.npz; done
  [ -s ${P}_ls_cfd_ia00.mp4 ] || python scripts/r8/r8_cf_detail.py --base ${P}_ls_pre.mp4 --cf ${P}_ls_raw.mp4 --lm $LMN --out ${P}_ls_cfd_ia00.mp4 --inner-alpha 0.0 | grep "^=="
  cp $LMN ${P}_ls_cfd_ia00.mediapipe.npz
  for lab in w05 w07 w10 ia00; do RW=${P}_ls_cfd_$lab.mp4; S=${RW%.mp4}_srt1.mp4; [ -s $S ] && continue
    gstep sr_$id$lab python scripts/r7/r7_layer_sr.py --ls-raw $RW --lm ${RW%.mp4}.mediapipe.npz --out $S --temporal 1 --batch 8 --mask down2 || rm -f $S
    cp ${RW%.mp4}.mediapipe.npz ${S%.mp4}.mediapipe.npz; done
done
touch /workspace/dub/r15f3/CF_GPU_DONE; echo QUEUE_DONE $(date +%T)
