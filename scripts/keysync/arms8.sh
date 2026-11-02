#!/bin/bash
cd /workspace/dub/keysync
for m in cond trunc:3 2way; do n=ig_${m/:/}; IDS="q18 q18" KS_DC=2:0 KS_IGUIDE=$m bash ks_fast.sh $n --cfg _s8 --vae-half --compile unet; done
echo ARMS8_DONE
