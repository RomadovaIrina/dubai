#!/bin/bash
# r12 KS speed arms, part 3: the VAE-decode and FP8 arms on m18 / m03 again after the ks_fast.py fixes (channels_last on the 2-D convs only; FP8 PerTensor)
cd /workspace/dub/keysync
until grep -q R12_SPEED2_DONE r12_speed2.log; do sleep 30; done
S="--cfg _s8 --vae-half --compile unet"
IDS="q18 q18 m03" KS_DC=2:0 KS_IGUIDE=trunc:3 bash ks_fast.sh f_t3dec $S --vae-cl --cudnn-bench --decoding-t 4
IDS="q18 q18 m03" KS_DC=2:0 KS_IGUIDE=trunc:3 bash ks_fast.sh f_t3fp8 $S --fp8
echo R12_SPEED3_DONE
