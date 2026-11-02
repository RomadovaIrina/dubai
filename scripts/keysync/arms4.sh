#!/bin/bash
cd /workspace/dub/keysync
IDS=q18 KS_DC=2:0 bash ks_fast.sh dc2s8 --cfg _s8
IDS=q18 KS_DC=2:0 bash ks_fast.sh dc2vh --vae-half
IDS="q18 q18" KS_DC=2:0 bash ks_fast.sh dc2cw --compile unet
echo ARMS4_DONE
