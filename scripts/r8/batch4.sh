#!/bin/bash
# r8-sep batch 4: noise-back clamped to its stationary level, 150-ms pause joins; source duck -3
cd /workspace/dub/sep; L=/workspace/dub/r8a/sep_logs
run() { bash scripts/r8/sep_mix.sh "$@" 01 02 03 04 05 > $L/$1.out 2>&1; echo "DONE $1 $(date +%T)" >> $L/batch.txt; }
run sepH8p htdemucs stems "--bed-pauses source --bed-noise-back on --src-duck-db -3" & run sepH8 htdemucs stems "--bed-noise-back on --src-duck-db -3" & wait
echo BATCH4_DONE >> $L/batch.txt
