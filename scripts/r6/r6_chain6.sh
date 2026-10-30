#!/bin/bash
# r6 audio queue 6: v5 = v4 with --dur-cost r6e + --fill-slowdown 0.9 (monotone drawl cost; 05 slow speaker), after v4
cd /workspace/dub/dubai
until grep -q "CHAIN5_DONE" /workspace/dub/artifacts/r6_chain5.log; do sleep 10; done
R6_QWEN=14b R6_EXTRA="--dur-cost r6e --dur-gap-qa-cost on --fill-slowdown 0.9" REMIX_ARGS="--src-duck-db -6 --unit-level on" bash scripts/r6/r6_audio.sh r6v5_14b
echo CHAIN6_DONE
