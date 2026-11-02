#!/bin/bash
# r12 KS speed arms, part 2: (a) the arms already evaluated without the lag fix are re-evaluated WITH it (ks_eval.sh default alpha 0.8; CPU + short SyncNet);
# (b) after r12_speed.sh: every arm on old03 (qo3) so the table covers old03 / m03 / m18
cd /workspace/dub/keysync
( for a in "f_base q18" "f_base m03" "f_t3 q18" "f_t3 m03" "f_t2 q18"; do bash ks_eval.sh $a | grep KS_EVAL; done; echo REEVAL_DONE ) > reeval.log 2>&1 &
until grep -q R12_SPEED_DONE r12_speed.log; do sleep 30; done
S="--cfg _s8 --vae-half --compile unet"
IDS=qo3 KS_DC=2:0 KS_IGUIDE=trunc:3 bash ks_fast.sh o_t3 $S
IDS=qo3 KS_DC=2:0 KS_IGUIDE=trunc:2 bash ks_fast.sh o_t2 $S
IDS=qo3 KS_DC=3:0 KS_IGUIDE=trunc:3 bash ks_fast.sh o_t3dc3 $S
IDS=qo3 KS_DC=2:0 KS_IGUIDE=trunc:3 bash ks_fast.sh o_t3dec $S --vae-cl --cudnn-bench --decoding-t 4
IDS=qo3 KS_DC=2:0 KS_IGUIDE=trunc:3 bash ks_fast.sh o_t3fp8 $S --fp8
bash ks_eval.sh f_base qo3 | grep KS_EVAL
wait; echo R12_SPEED2_DONE
