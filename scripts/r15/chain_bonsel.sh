#!/bin/bash
# r15 W1c: SELECTIVE best-of-3 (LS_BON_SEL=1: only lazy / smeared windows try other noises, strong continuity, sticky noise) on s4p1500c8
cd /workspace/dub/dubai; X=/workspace/dub/r15w/eval; CK=/workspace/dub/r15w/ckpt/r2_s4_step_01500_ema.pt
for id in ${*:-old03 m03 m18}; do
  echo "== $id bonsel $(date +%T)"; rm -f $X/bonsel_$id.bon.jsonl
  R10_GEN_PY=scripts/r13/distill/student_gen.py R13_STUDENT_LORA=$CK R13_STUDENT_STEPS=4 LS_STRIDE=8 LS_OVERLAP=continue LS_BON=3 LS_BON_SEL=1 BON_LOG=$X/bonsel_$id.bon.jsonl \
    V=bonsel bash scripts/r15/arm.sh $id 2>&1 | grep "^=="
  grep -h "bon-sel" /workspace/dub/artifacts/r4m/r9g_q2_${id}_bonsel/*_bonsel.log | tail -3
  R=s4c8; [ $id = m03 ] && R=s4c8u
  bash scripts/r15/pair.sh $id $R bonsel "student s4p1500c8" "selective best-of-3" > /dev/null 2>&1
done
echo BONSEL_DONE
