#!/bin/bash
# the candidate safe stack on all 5: one process + fast T3 + clause lazy cfg (no plan-top, no pipeline candidate cut), gated vs r8e2e0b
cd /workspace/dub/dubai
echo "STACK r8fast4 start $(date +%T)"
env R8_ONEPROC=on R8_PIPE_EXTRA="--tts-fast-t3 on" DUB_TTS_FAST_T3=1 R8_CLAUSE_ARGS="--lazy-cfg on" bash scripts/r8/r8_audio.sh r8fast4 01 02 03 04 05 > /workspace/dub/r8a/r8fast4.log 2>&1
echo "STACK r8fast4 done $(date +%T) $(grep R8_AUDIO_DONE /workspace/dub/r8a/r8fast4.log)"
echo LEVERS6_DONE
