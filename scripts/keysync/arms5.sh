#!/bin/bash
cd /workspace/dub/keysync
bash ks_fast.sh base4d
IDS=q18 KS_DC=2:0 bash ks_fast.sh f_dc2
IDS=q18 KS_DC=2:0 bash ks_fast.sh f_dc2vh --vae-half
IDS=q18 KS_DC=2:0 bash ks_fast.sh f_dc2s8 --cfg _s8 --vae-half
IDS="q18 q18" KS_DC=2:0 bash ks_fast.sh f_dc2cw --compile unet --vae-half
echo ARMS5_DONE
