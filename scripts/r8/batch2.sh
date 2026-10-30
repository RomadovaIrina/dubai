#!/bin/bash
# r8-sep batch 2: the Spleeter-era workarounds relaxed on the htdemucs bed (pauses = source), Spleeter relaxed as the control
cd /workspace/dub/sep; L=/workspace/dub/r8a/sep_logs
run() { bash scripts/r8/sep_mix.sh "$@" 01 02 03 04 05 > $L/$1.out 2>&1; echo "DONE $1 $(date +%T)" >> $L/batch.txt; }
P="--bed-pauses source"
run sepH1p htdemucs stems "$P --src-duck-db 0" & run sepH1 htdemucs stems "--src-duck-db 0" & wait
run sepH2p htdemucs stems "$P --src-duck-db 0 --gap-clean-nr 0 --gap-clean-nr-ns 0" & run sepH3p htdemucs stems "$P --src-duck-db 0 --gap-clean-nr 0 --gap-clean-nr-ns 0 --room-fill off" & wait
run sepH4p htdemucs stems "$P --src-duck-db 0 --gap-clean off --gap-bed-cap-db 0 --room-fill off --residue-db 0" & run sepS1 spleeter stems "--src-duck-db 0 --gap-clean-nr 0 --gap-clean-nr-ns 0" & wait
echo BATCH2_DONE >> $L/batch.txt
