#!/bin/bash
# r15 W1 + W2 chain on the held-out videos (each on its q2 + F3 baseline's dub, the q2 + F3 composite):
#   bon3  = s4p1500c8 + LS_BON=3 (best-of-3 per window, sequential pick; scripts/r15/bon.py)
#   kf2 / kf3 = s4p1500c8 + LS_KF=2 / 3 (keyframes + DIS interpolation in the aligned crop space; scripts/r15/kf.py)
#   tkf2  = the q2 TEACHER + LS_KF=2 (the teacher's quality at ~half its generator cost?)
#   bash scripts/r15/chain_w12.sh [old03 m03 m18]     ARMS="bon3 kf2 kf3 tkf2" to choose
set -u
cd /workspace/dub/dubai; X=/workspace/dub/r15w/eval; CK=/workspace/dub/r15w/ckpt/r2_s4_step_01500_ema.pt
ST="R10_GEN_PY=scripts/r13/distill/student_gen.py R13_STUDENT_LORA=$CK R13_STUDENT_STEPS=4 LS_STRIDE=8 LS_OVERLAP=continue"
for id in ${*:-old03 m03 m18}; do
  until [ -s $X/q2_$id/q2_$id.mp4 ] || { grep -q BASE_DONE /workspace/dub/r15w/base_q2b.log 2>/dev/null && [ "$id" != old03 ]; }; do sleep 20; done
  [ -s $X/q2_$id/q2_$id.mp4 ] || { echo "no baseline for $id: skipped"; continue; }
  for arm in ${ARMS:-bon3 kf2 kf3 tkf2}; do
    echo "== $id $arm $(date +%T)"
    case $arm in
      bon3) env $ST LS_BON=3 BON_LOG=$X/bon3_$id.bon.jsonl V=bon3 bash scripts/r15/arm.sh $id;;
      kf2)  env $ST LS_KF=2 V=kf2 bash scripts/r15/arm.sh $id;;
      kf3)  env $ST LS_KF=3 V=kf3 bash scripts/r15/arm.sh $id;;
      tkf2) env LS_KF=2 V=tkf2 bash scripts/r15/arm.sh $id;;
    esac 2>&1 | grep "^=="
    grep -h "\[bon\]\|\[kf\]" $X/${arm}_$id/logs/face_gen.log /workspace/dub/artifacts/r4m/r9g_q2_${id}_$arm/*.log 2>/dev/null | tail -2
  done
  bash scripts/r15/metrics.sh $id q2 s4c8 tb ${ARMS:-bon3 kf2 kf3 tkf2} > /dev/null 2>&1; cat $X/metrics_$id.txt
  for arm in ${ARMS:-bon3 kf2 kf3}; do [ $arm = tkf2 ] && continue; bash scripts/r15/pair.sh $id s4c8 $arm "student s4p1500c8" "$arm" > /dev/null 2>&1; done
  [[ " ${ARMS:-tkf2} " == *" tkf2 "* ]] && bash scripts/r15/pair.sh $id q2 tkf2 "q2 + F3 (production)" "q2 teacher, keyframes /2 + interp" > /dev/null 2>&1
  echo "== $id W12 done $(date +%T)"
done
echo W12_CHAIN_DONE
