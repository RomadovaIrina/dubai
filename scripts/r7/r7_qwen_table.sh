#!/bin/bash
# r7: 7B Q8_0 vs 14B Q5_K_M on the r6 duration-aware pipeline (scripts/r6/r6_audio.sh), all five videos, ONE video at a time (clean timings / VRAM).
# The faithfulness judge (r5_faith) is the 14B for every arm. Then scripts/r7/r7_qwen_table.py builds the table.
#     bash scripts/r7/r7_qwen_table.sh [arms]      arms: q14 q7 q7j14 q14r1
set -u; cd ${DUB_ROOT:-/workspace/dub/dubai}
for arm in ${*:-q14 q7 q7j14 q14r1}; do
  case $arm in
    q14) env PAR=1 R6_QWEN=14b bash scripts/r6/r6_audio.sh $arm ;;
    q7) env PAR=1 R6_QWEN=7b bash scripts/r6/r6_audio.sh $arm ;;
    q7j14) env PAR=1 R6_QWEN=7b DUB_QWEN_JUDGE=14b bash scripts/r6/r6_audio.sh $arm ;;
    q14r1) env PAR=1 R6_QWEN=14b R6_EXTRA="--dur-rounds 1" bash scripts/r6/r6_audio.sh $arm ;;
  esac
  echo "ARM_DONE $arm $(date +%T)"
done
