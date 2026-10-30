#!/bin/bash
# r8 speed levers, each ALONE on 04 + 05 from the raw input (baseline = r8e2e0 / r8e2e0b), one after the other under gpu.lock
cd /workspace/dub/dubai; D=" --short-s 0.3 --budget-scales 1.15,1.3,1.5,1.75 --cap-scale 1.2 --min-piece 0.2 --max-tts 4 --natural-cuts on --fillers on"
run() { t=$1; shift; echo "LEVER $t start $(date +%T)"; env "$@" bash scripts/r8/r8_audio.sh $t 04 05 > /workspace/dub/r8a/$t.log 2>&1; echo "LEVER $t done $(date +%T) $(grep R8_AUDIO_DONE /workspace/dub/r8a/$t.log)"; }
run r8sL1 R8_PIPE_EXTRA="--dur-top 1 --dur-rounds 1"
run r8sL2 R8_PIPE_EXTRA="--tts-fast-t3 on" DUB_TTS_FAST_T3=1
run r8sL3 R8_RETAKE_ARGS="$D --max-tts 2" R8_CLAUSE_ARGS="--lazy-cfg on --max-cuts 2"
echo LEVERS1_DONE
