#!/bin/bash
cd /workspace/dub/keysync
IDS="q18 q18" KS_DC=2:0 KS_IGUIDE=trunc:2 bash ks_fast.sh ig2_trunc2 --cfg _s8 --vae-half --compile unet
IDS="q18 q18" KS_DC=3:0 KS_IGUIDE=trunc:3 bash ks_fast.sh ig2_t3dc3 --cfg _s8 --vae-half --compile unet
IDS="q18 q18" KS_DC=2:0 KS_IGUIDE=trunc:3 bash ks_fast.sh ig2_t3vae --cfg _s8 --vae-half --compile unet --vae-cl --cudnn-bench --decoding-t 4
echo ARMS11_DONE
