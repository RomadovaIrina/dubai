#!/bin/bash
# r15 W: m03 with the face-units fix (0009, LS_FACE_UNITS=1 default): q2u (teacher q2), s4c8u (student), tkf2u (teacher + keyframes / 2),
# kfv3u (student + voiced-adaptive keyframes / 3), face stages on the q2_m03 dub; then the eye clips vs q2u.
cd /workspace/dub/dubai; X=/workspace/dub/r15w/eval; CK=/workspace/dub/r15w/ckpt/r2_s4_step_01500_ema.pt
ST="R10_GEN_PY=scripts/r13/distill/student_gen.py R13_STUDENT_LORA=$CK R13_STUDENT_STEPS=4 LS_STRIDE=8 LS_OVERLAP=continue"
for arm in q2u tkf2u s4c8u kfv3u; do
  echo "== m03 $arm $(date +%T)"
  case $arm in
    q2u)   V=q2u bash scripts/r15/arm.sh m03;;
    tkf2u) LS_KF=2 V=tkf2u bash scripts/r15/arm.sh m03;;
    s4c8u) env $ST V=s4c8u bash scripts/r15/arm.sh m03;;
    kfv3u) env $ST LS_KF=3 KF_VOICED=1 V=kfv3u bash scripts/r15/arm.sh m03;;
  esac 2>&1 | grep "^=="
done
bash scripts/r15/pair.sh m03 q2u tkf2u "q2 + F3 (production, face units)" "teacher, keyframes /2 + interp" > /dev/null 2>&1
bash scripts/r15/pair.sh m03 s4c8u tkf2u "student s4p1500c8 (face units)" "teacher, keyframes /2 + interp" > /dev/null 2>&1
bash scripts/r15/pair.sh m03 q2u kfv3u "q2 + F3 (production, face units)" "student, voiced-adaptive keyframes /3" > /dev/null 2>&1
echo FU_CHAIN_DONE
