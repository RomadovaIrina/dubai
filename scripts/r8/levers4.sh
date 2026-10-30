#!/bin/bash
# the stack on all 5 from the raw input: L2 + L3 + L4 + L5 + L6 (L1 is moot without the pipeline duration candidates)
cd /workspace/dub/dubai; D=" --short-s 0.3 --budget-scales 1.15,1.3,1.5,1.75 --cap-scale 1.2 --min-piece 0.2 --max-tts 4 --natural-cuts on --fillers on"
echo "STACK start $(date +%T)"
env R8_PIPE_EXTRA="--dur-translate off --tts-fast-t3 on" DUB_TTS_FAST_T3=1 R8_ONEPROC=on R8_RETAKE_ARGS="$D --max-tts 2" R8_CLAUSE_ARGS="--lazy-cfg on --max-cuts 2 --plan-top 2" \
  bash scripts/r8/r8_audio.sh r8fast1 01 02 03 04 05 > /workspace/dub/r8a/r8fast1.log 2>&1
echo "STACK done $(date +%T) $(grep R8_AUDIO_DONE /workspace/dub/r8a/r8fast1.log)"
