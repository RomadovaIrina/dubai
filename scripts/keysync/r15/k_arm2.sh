#!/bin/bash
# r15 K: k_arm.sh with the evals in the background (their SyncNet calls interleave on gpu.lock with the next GPU job)
A=$1; shift; K=/workspace/dub/keysync; cd $K
IDS="${IDS:-wu q18 m03 qo3}" KS_NOEVAL=1 KS_LMK_CACHE=$K/lmk bash ks_fast.sh $A "$@"
(for i in ${EVAL_IDS:-q18 m03 qo3}; do bash k_eval.sh $A $i | grep K_EVAL; done > fast/$A/evals.log 2>&1) &
