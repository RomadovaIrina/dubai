#!/bin/bash
# r15 W2c: the teacher q2 + keyframes / 2 + c8 (LS_STRIDE=8 LS_OVERLAP=continue, eye-preferred on the teacher in r14) = tkf2c8 on old03 / m03 / m18
# (m03 with the face-units fix, now default), then the eye clips vs q2 (m03: vs q2u) and the q2 | kfv3 clips the coordinator asked for.
cd /workspace/dub/dubai; X=/workspace/dub/r15w/eval
for id in old03 m03 m18; do
  echo "== $id tkf2c8 $(date +%T)"; LS_KF=2 LS_STRIDE=8 LS_OVERLAP=continue V=tkf2c8 bash scripts/r15/arm.sh $id 2>&1 | grep "^=="
  R=q2; [ $id = m03 ] && R=q2u
  bash scripts/r15/pair.sh $id $R tkf2c8 "q2 + F3 (production)" "teacher + c8, keyframes /2 + interp" > /dev/null 2>&1
done
for id in old03 m18; do bash scripts/r15/pair.sh $id q2 kfv3 "q2 + F3 (production)" "student, voiced-adaptive keyframes /3" > /dev/null 2>&1; done
echo TKF_DONE
