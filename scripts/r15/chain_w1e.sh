#!/bin/bash
# r15 W1b: best-of-3 with the EARLY pick (LS_BON_EARLY=1: candidates run the first step only, the winner the rest) on s4p1500c8
cd /workspace/dub/dubai; X=/workspace/dub/r15w/eval; CK=/workspace/dub/r15w/ckpt/r2_s4_step_01500_ema.pt
for id in ${*:-old03 m03 m18}; do
  until [ -s $X/q2_$id/q2_$id.mp4 ] || { grep -q BASE_DONE /workspace/dub/r15w/base_q2b.log 2>/dev/null && [ "$id" != old03 ]; }; do sleep 20; done
  [ -s $X/q2_$id/q2_$id.mp4 ] || continue
  echo "== $id bon3e $(date +%T)"
  R10_GEN_PY=scripts/r13/distill/student_gen.py R13_STUDENT_LORA=$CK R13_STUDENT_STEPS=4 LS_STRIDE=8 LS_OVERLAP=continue LS_BON=3 LS_BON_EARLY=1 BON_LOG=$X/bon3e_$id.bon.jsonl \
    V=bon3e bash scripts/r15/arm.sh $id 2>&1 | grep "^=="
  bash scripts/r15/pair.sh $id s4c8 bon3e "student s4p1500c8" "best-of-3 early pick" > /dev/null 2>&1
done
echo W1E_CHAIN_DONE
