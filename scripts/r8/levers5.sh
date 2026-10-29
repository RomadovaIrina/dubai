#!/bin/bash
# quality-safe stacks on all 5 from the raw input, gated vs the from-scratch baseline r8e2e0b
cd /workspace/dub/dubai
run() { t=$1; shift; echo "STACK $t start $(date +%T)"; env "$@" bash scripts/r8/r8_audio.sh $t 01 02 03 04 05 > /workspace/dub/r8a/$t.log 2>&1; echo "STACK $t done $(date +%T) $(grep R8_AUDIO_DONE /workspace/dub/r8a/$t.log)"; }
run r8fast2 R8_ONEPROC=on R8_PIPE_EXTRA="--dur-top 1 --dur-rounds 1" R8_CLAUSE_ARGS="--lazy-cfg on --plan-top 4"
run r8fast3 R8_ONEPROC=on R8_PIPE_EXTRA="--dur-top 1 --dur-rounds 1 --tts-fast-t3 on" DUB_TTS_FAST_T3=1 R8_CLAUSE_ARGS="--lazy-cfg on --plan-top 4"
echo LEVERS5_DONE
