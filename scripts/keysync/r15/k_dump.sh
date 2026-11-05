#!/bin/bash
# k_dump.sh <jobs>: ks_dump.py under gpu.lock
K=/workspace/dub/keysync; cd $K/code; . $K/venv/bin/activate
flock /workspace/dub/gpu.lock python $K/ks_dump.py --jobs $1 --out $K/dump > $1.log 2>&1; grep -E "DUMP|Error|error" $1.log | tail -12
