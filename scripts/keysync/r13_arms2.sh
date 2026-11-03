#!/bin/bash
# r13 arms rerun after the fixes (vae-cl: own 4-D weights only; fp8: half-bias Linear only); queued after r13_arms.sh
cd /workspace/dub/keysync; until grep -q R13_ARMS_DONE r13_arms.log; do sleep 60; done
S="--cfg _s8 --vae-half --compile unet"; export IDS="q18 q18 m03 qo3" KS_NOEVAL=1
for a in f_s8dec f_s8fp8; do mv fast/$a fast/${a}_crash1 2>/dev/null; done
ev() { ( for i in q18 m03 qo3; do bash ks_eval.sh $1 $i | grep KS_EVAL; done ) & }
KS_DC=2:0 bash ks_fast.sh f_s8dec $S --vae-cl --cudnn-bench --decoding-t 4; ev f_s8dec
KS_DC=2:0 bash ks_fast.sh f_s8fp8 $S --fp8; grep -a FP8_LAYERS fast/f_s8fp8/run.log | head -2; ev f_s8fp8
wait; echo R13_ARMS2_DONE
