#!/bin/bash
for p in $(pgrep -f "ks_plain.py --ks"); do kill $p; done; for p in $(pgrep -f "bash k_plain.sh"); do kill $p; done; sleep 1; pgrep -fc "ks_plain.py --ks"
