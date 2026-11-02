#!/bin/bash
cd /workspace/dub/keysync
IDS=q18 KS_DC=2:0 bash ks_fast.sh g_dc2vh --vae-half
IDS="q18 q18 qo3" KS_DC=2:0 bash ks_fast.sh g_best --cfg _s8 --vae-half --compile unet
echo ARMS6_DONE
