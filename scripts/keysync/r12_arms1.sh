#!/bin/bash
# r12 KS: source-audio A/B on old03 (no dub audio on r12-k yet): baseline, the r11 best stack, + trunc:3
cd /workspace/dub/keysync; export KS_DUB=$PWD/lm/qo3_src16k.wav KS_MIX=$PWD/lm/qo3_src.m4a KS_LM=$PWD/lm/qo3_lm.npz IDS=qo3
bash ks_fast.sh r12_base
KS_DC=2:0 bash ks_fast.sh r12_stack_warm --cfg _s8 --vae-half --compile unet
KS_DC=2:0 bash ks_fast.sh r12_stack --cfg _s8 --vae-half --compile unet
KS_DC=2:0 KS_IGUIDE=trunc:3 bash ks_fast.sh r12_t3 --cfg _s8 --vae-half --compile unet
echo R12_ARMS1_DONE
