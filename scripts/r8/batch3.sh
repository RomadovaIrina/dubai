#!/bin/bash
# r8-sep batch 3: the vocals stem noise back in the bed (02), source duck 0 / -3, denoise halved
cd /workspace/dub/sep; L=/workspace/dub/r8a/sep_logs
run() { bash scripts/r8/sep_mix.sh "$@" 01 02 03 04 05 > $L/$1.out 2>&1; echo "DONE $1 $(date +%T)" >> $L/batch.txt; }
P="--bed-pauses source --bed-noise-back on"
run sepH5p htdemucs stems "$P --src-duck-db 0" & run sepH5 htdemucs stems "--bed-noise-back on --src-duck-db 0" & wait
run sepH6p htdemucs stems "$P --src-duck-db -3" & run sepH7p htdemucs stems "$P --src-duck-db 0 --gap-clean-nr 0.3 --gap-clean-nr-ns 0.35" & wait
echo BATCH3_DONE >> $L/batch.txt
