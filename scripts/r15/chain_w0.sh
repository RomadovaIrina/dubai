#!/bin/bash
# r15 W0 chain: per held-out video (as soon as its q2 + F3 baseline exists): TBDub arm (tb), the s4p1500c8 student arm (s4c8), metrics,
# the eye clips (q2 | tb, s4c8 | tb). Every GPU step queues on gpu.lock inside the stages.
#   bash scripts/r15/chain_w0.sh [old03 m03 m18]
set -u
cd /workspace/dub/dubai; X=/workspace/dub/r15w/eval; CK=/workspace/dub/r15w/ckpt/r2_s4_step_01500_ema.pt
for id in ${*:-old03 m03 m18}; do
  until [ -s $X/q2_$id/q2_$id.mp4 ] || { grep -q BASE_DONE /workspace/dub/r15w/base_q2b.log 2>/dev/null && [ "$id" != old03 ]; }; do sleep 20; done
  [ -s $X/q2_$id/q2_$id.mp4 ] || { echo "no baseline for $id: skipped"; continue; }
  echo "== $id tb $(date +%T)"; V=tb R10_GEN_PY=scripts/r15/tbdub_gen.py bash scripts/r15/arm.sh $id 2>&1 | grep "^=="
  echo "== $id s4c8 $(date +%T)"; V=s4c8 R10_GEN_PY=scripts/r13/distill/student_gen.py R13_STUDENT_LORA=$CK R13_STUDENT_STEPS=4 LS_STRIDE=8 LS_OVERLAP=continue \
    bash scripts/r15/arm.sh $id 2>&1 | grep "^=="
  bash scripts/r15/metrics.sh $id q2 s4c8 tb > /dev/null 2>&1; cat $X/metrics_$id.txt
  bash scripts/r15/pair.sh $id q2 tb "q2 + F3 (production)" "TBDub student 2-step" > /dev/null 2>&1
  bash scripts/r15/pair.sh $id s4c8 tb "student s4p1500c8" "TBDub student 2-step" > /dev/null 2>&1
  echo "== $id W0 done $(date +%T)"
done
echo W0_CHAIN_DONE
