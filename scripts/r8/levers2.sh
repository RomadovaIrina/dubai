#!/bin/bash
# L4 alone on 04 + 05: the pipeline without its own duration-aware candidates (retake / clause takes / burst_align do the timing after it)
cd /workspace/dub/dubai
run() { t=$1; shift; echo "LEVER $t start $(date +%T)"; env "$@" bash scripts/r8/r8_audio.sh $t 04 05 > /workspace/dub/r8a/$t.log 2>&1; echo "LEVER $t done $(date +%T) $(grep R8_AUDIO_DONE /workspace/dub/r8a/$t.log)"; }
run r8sL4 R8_PIPE_EXTRA="--dur-translate off"
echo LEVERS2_DONE
