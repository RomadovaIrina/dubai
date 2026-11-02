#!/bin/bash
cd /workspace/dub/keysync
IDS=q18 KS_DC=2:0 bash ks_fast.sh dc2s6 --cfg _s6
IDS=q18 KS_DC=2:0 bash ks_fast.sh dc2c --compile unet
echo ARMS3_DONE
