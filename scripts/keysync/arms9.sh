#!/bin/bash
cd /workspace/dub/keysync
for m in ucc mix:3 trunc:6; do n=ig_${m/:/}; IDS="q18 q18" KS_DC=2:0 KS_IGUIDE=$m bash ks_fast.sh $n --cfg _s8 --vae-half --compile unet; done
echo ARMS9_DONE
