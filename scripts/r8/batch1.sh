#!/bin/bash
# r8-sep batch 1: the r8v4 mix on each candidate bed, workarounds unchanged (2 tags at a time on the CPU; the gap ASR queues on gpu.lock)
cd /workspace/dub/sep; L=/workspace/dub/r8a/sep_logs
run() { bash scripts/r8/sep_mix.sh "$@" 01 02 03 04 05 > $L/$1.out 2>&1; echo "DONE $1 $(date +%T)" >> $L/batch.txt; }
run sepH0 htdemucs stems "" & run sepH0r htdemucs residual "" & wait
run sepF0r htdemucs_ft residual "" & run sepH0p htdemucs stems "--bed-pauses source" & wait
run sepS0 spleeter stems "" & run sepF0 htdemucs_ft stems "" & wait
echo BATCH1_DONE >> $L/batch.txt
