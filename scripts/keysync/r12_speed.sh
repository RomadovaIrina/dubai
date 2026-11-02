#!/bin/bash
# r12 KS speed arms (dub voice): the 10-step finals = the baseline; the r11 stack (8 steps + DC 2:0 + fp16 VAE + compile) + each lever; m18 (warm, twice) + m03
cd /workspace/dub/keysync; until grep -q LAGFIX_DONE lagfix.log; do sleep 15; done
for i in q18 m03 qo3; do bash ks_eval.sh f_base $i | grep KS_EVAL; done
S="--cfg _s8 --vae-half --compile unet"
IDS="q18 q18 m03" KS_DC=2:0 KS_IGUIDE=trunc:3 bash ks_fast.sh f_t3 $S
IDS="q18 q18 m03" KS_DC=2:0 KS_IGUIDE=trunc:2 bash ks_fast.sh f_t2 $S
IDS="q18 q18 m03" KS_DC=3:0 KS_IGUIDE=trunc:3 bash ks_fast.sh f_t3dc3 $S
IDS="q18 q18 m03" KS_DC=2:0 KS_IGUIDE=trunc:3 bash ks_fast.sh f_t3dec $S --vae-cl --cudnn-bench --decoding-t 4
IDS="q18 q18 m03" KS_DC=2:0 KS_IGUIDE=trunc:3 bash ks_fast.sh f_t3fp8 $S --fp8
echo R12_SPEED_DONE
