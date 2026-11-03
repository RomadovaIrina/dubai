#!/bin/bash
# r13 FP8 arm, take 2: UNets cast to bf16, float8 PerRow on the 16-aligned Linear layers; m18 first (cold + warm), the rest only if it runs and holds sync
cd /workspace/dub/keysync; S="--cfg _s8 --vae-half --compile unet"; export KS_NOEVAL=1 KS_FP8_BF16=1 KS_AC_BF16=1
IDS="q18 q18" KS_DC=2:0 bash ks_fast.sh f_s8fp8d $S --fp8; grep -a FP8_LAYERS fast/f_s8fp8d/run.log | head -2
grep -a -h KS_JOB fast/f_s8fp8d/run.log | sed -E "s/\"stages_s\": \{[^}]*\}//" | cut -c1-160; grep -a "Error" fast/f_s8fp8d/run.log | tail -2 | cut -c1-200
bash ks_eval.sh f_s8fp8d q18 | grep KS_EVAL; echo FP8B_DONE
