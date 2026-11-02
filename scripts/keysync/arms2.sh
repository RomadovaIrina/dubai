#!/bin/bash
cd /workspace/dub/keysync
IDS=q18 KS_DC=2:0 bash ks_fast.sh dc2b0
IDS=q18 KS_ICHUNK=6 bash ks_fast.sh ich6
IDS=q18 KS_DC=3:0 bash ks_fast.sh dc3b0
IDS=q18 bash ks_fast.sh dt7 --decoding-t 7
echo ARMS2_DONE
