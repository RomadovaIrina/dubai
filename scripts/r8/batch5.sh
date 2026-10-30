#!/bin/bash
cd /workspace/dub/sep; L=/workspace/dub/r8a/sep_logs
run() { bash scripts/r8/sep_mix.sh "$@" 01 02 03 04 05 > $L/$1.out 2>&1; echo "DONE $1 $(date +%T)" >> $L/batch.txt; }
run sepH9p htdemucs stems "--bed-pauses source --bed-noise-back on --src-duck-db 0"
echo BATCH5_DONE >> $L/batch.txt
