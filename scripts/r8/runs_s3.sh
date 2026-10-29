#!/bin/bash
# sequential upstream draws for the envelope gate: draw 2 (seeds +1) on 01-03 (04 / 05 exist), draw 3 (seeds +2) on all 5
cd /workspace/dub/dubai; D=" --short-s 0.3 --budget-scales 1.15,1.3,1.5,1.75 --cap-scale 1.2 --min-piece 0.2 --max-tts 4 --natural-cuts on --fillers on"
echo "r8s2b start $(date +%T)"; env R8_PIPE_EXTRA="--seed 1248" R8_RETAKE_ARGS="$D --seed 8" R8_CLAUSE_ARGS="--seed 12" bash scripts/r8/r8_audio.sh r8s2b 01 02 03 > /workspace/dub/r8a/r8s2b.log 2>&1; echo "r8s2b done $(date +%T)"
echo "r8s3 start $(date +%T)"; env R8_PIPE_EXTRA="--seed 1249" R8_RETAKE_ARGS="$D --seed 9" R8_CLAUSE_ARGS="--seed 13" bash scripts/r8/r8_audio.sh r8s3 01 02 03 04 05 > /workspace/dub/r8a/r8s3.log 2>&1; echo "r8s3 done $(date +%T)"
echo RUNS_S3_DONE
