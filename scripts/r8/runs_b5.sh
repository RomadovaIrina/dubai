#!/bin/bash
# 1. the sequential baseline with every seed +1 on 04 + 05 (the draw-to-draw spread of upstream), 2. the batched build on all 5
cd /workspace/dub/dubai; D=" --short-s 0.3 --budget-scales 1.15,1.3,1.5,1.75 --cap-scale 1.2 --min-piece 0.2 --max-tts 4 --natural-cuts on --fillers on"
echo "r8s2 start $(date +%T)"; env R8_PIPE_EXTRA="--seed 1248" R8_RETAKE_ARGS="$D --seed 8" R8_CLAUSE_ARGS="--seed 12" bash scripts/r8/r8_audio.sh r8s2 04 05 > /workspace/dub/r8a/r8s2.log 2>&1; echo "r8s2 done $(date +%T)"
echo "r8b5 start $(date +%T)"; env DUB_TTS_BATCH=1 bash scripts/r8/r8_audio.sh r8b5 01 02 03 04 05 > /workspace/dub/r8a/r8b5.log 2>&1; echo "r8b5 done $(date +%T)"
echo RUNS_B5_DONE
