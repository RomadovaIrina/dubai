#!/bin/bash
cd /workspace/dub/keysync
for i in q18 qo3; do bash ks_eval.sh base $i | grep KS_EVAL; done
IDS=q18 bash ks_fast.sh dt14 --decoding-t 14
IDS=q18 bash ks_fast.sh s6 --cfg _s6
IDS=q18 KS_DC=2:0 bash ks_fast.sh dc2b0
IDS=q18 KS_DC=3:0 bash ks_fast.sh dc3b0
IDS=q18 KS_DC=2:2 bash ks_fast.sh dc2b2
echo ARMS1_DONE
