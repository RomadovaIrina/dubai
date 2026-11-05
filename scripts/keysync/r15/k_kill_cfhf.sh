#!/bin/bash
# stop the cfhf finals (scripts only; the patterns live here, not on the caller command line)
for p in $(pgrep -f "bash k_cfhf"); do kill $p; done
for p in $(pgrep -f "bash ks_final.sh"); do kill $p; done
sleep 1; for p in $(pgrep -f "ks_layer.py|ks_fix.py|ks_tsmooth.py|ks_paste.py|mp_landmarks.py"); do kill $p; done
sleep 1; ps aux | grep -E "ks_final|ks_fix|ks_layer|k_cfhf" | grep -v grep | wc -l
