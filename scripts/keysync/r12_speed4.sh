#!/bin/bash
# r12 KS speed arms, part 4 (coordinator): the r11 stack WITHOUT truncation (8 steps + DC2:0 + fp16 VAE + compile + full 3-way CFG) on m18 / m03 / old03,
# then 6 steps + DC2 full CFG, then the decode / FP8 fixes stacked on the full-CFG 8-step stack. Same eval (SyncNet, flashes, jerk; alpha 0.8 smoothing).
cd /workspace/dub/keysync
( bash ks_eval.sh f_base qo3 | grep KS_EVAL ) &
while pgrep -f "ks_fast\.sh" > /dev/null; do sleep 20; done
IDS="q18 q18 m03 qo3" KS_DC=2:0 bash ks_fast.sh f_s8 --cfg _s8 --vae-half --compile unet
IDS="q18 q18 m03 qo3" KS_DC=2:0 bash ks_fast.sh f_s6 --cfg _s6 --vae-half --compile unet
IDS="q18 q18 m03 qo3" KS_DC=2:0 bash ks_fast.sh f_s8dec --cfg _s8 --vae-half --compile unet --vae-cl --cudnn-bench --decoding-t 4
IDS="q18 q18 m03 qo3" KS_DC=2:0 bash ks_fast.sh f_s8fp8 --cfg _s8 --vae-half --compile unet --fp8
wait; echo R12_SPEED4_DONE
