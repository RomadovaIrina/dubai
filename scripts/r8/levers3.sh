#!/bin/bash
# L5 (one process, models loaded once) and L6 (clause layouts planned on estimated lengths, top 2 synthesised), each alone on 04 + 05.
# The new code is put in place here, under gpu.lock, so no other audio run is mid-way through these scripts.
cd /workspace/dub/dubai
cp /workspace/dub/r8a/r8_audio.sh scripts/r8/r8_audio.sh && cp /tmp/p20/scripts/r8/clause_takes.py scripts/r8/clause_takes.py && echo "code in place $(date +%T)"
run() { t=$1; shift; echo "LEVER $t start $(date +%T)"; env "$@" bash scripts/r8/r8_audio.sh $t 04 05 > /workspace/dub/r8a/$t.log 2>&1; echo "LEVER $t done $(date +%T) $(grep R8_AUDIO_DONE /workspace/dub/r8a/$t.log)"; }
run r8sL5 R8_ONEPROC=on
run r8sL6 R8_CLAUSE_ARGS="--plan-top 2"
echo LEVERS3_DONE
