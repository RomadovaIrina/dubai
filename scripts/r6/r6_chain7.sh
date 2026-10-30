#!/bin/bash
# r6 audio queue 7: v6 = v2 (r6c, gap-QA cost guard, fill 0.95) + --close-inside-pauses on + remix --unit-level on, 14B; after v5
cd /workspace/dub/dubai
until grep -q "CHAIN6_DONE" /workspace/dub/artifacts/r6_chain6.log; do sleep 10; done
R6_QWEN=14b R6_EXTRA="--dur-cost r6c --dur-gap-qa-cost on --close-inside-pauses on" REMIX_ARGS="--src-duck-db -6 --unit-level on" bash scripts/r6/r6_audio.sh r6v6_14b
echo CHAIN7_DONE
