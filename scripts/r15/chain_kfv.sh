#!/bin/bash
# r15 W2b: voiced-adaptive keyframes (LS_KF=3 KF_VOICED=1: dense on dub-voiced frames, keyframes / 3 + DIS in crop space on dub-silent ones) on s4p1500c8
cd /workspace/dub/dubai; X=/workspace/dub/r15w/eval; CK=/workspace/dub/r15w/ckpt/r2_s4_step_01500_ema.pt
for id in ${*:-old03 m03 m18}; do
  echo "== $id kfv3 $(date +%T)"
  R10_GEN_PY=scripts/r13/distill/student_gen.py R13_STUDENT_LORA=$CK R13_STUDENT_STEPS=4 LS_STRIDE=8 LS_OVERLAP=continue LS_KF=3 KF_VOICED=1 \
    V=kfv3 bash scripts/r15/arm.sh $id 2>&1 | grep "^=="
  grep -h "\[kf\]" /workspace/dub/artifacts/r4m/r9g_q2_${id}_kfv3/*_kfv3.log | awk "{k+=\$3; n+=\$6} END {print \"kfv3 $id keyframes\", k, \"of\", n}"
  bash scripts/r15/pair.sh $id s4c8 kfv3 "student s4p1500c8" "voiced-adaptive keyframes /3" > /dev/null 2>&1
done
echo KFV_CHAIN_DONE
