#!/bin/bash
# r15 K: one arm = the resident KeySync worker on a 4 s warm-up clip + q18 m03 qo3 (env passes through: KS_DC, KS_ICHUNK, ...), then k_eval.sh per video.
# usage: bash k_arm.sh <arm> [ks_fast.py args]
A=$1; shift; K=/workspace/dub/keysync; cd $K
IDS="${IDS:-wu q18 m03 qo3}" KS_NOEVAL=1 KS_LMK_CACHE=$K/lmk bash ks_fast.sh $A "$@"
for i in q18 m03 qo3; do bash k_eval.sh $A $i | grep K_EVAL; done; echo K_ARM_DONE $A
