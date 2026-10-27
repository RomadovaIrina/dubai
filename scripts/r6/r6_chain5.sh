#!/bin/bash
# r6 audio queue 5: v4 = v2 (r6c, gap-QA cost guard, fill 0.95) + --dur-cost r6d (inside-phrase pauses of the take) + remix --unit-level on, 14B
cd /workspace/dub/dubai
R6_QWEN=14b R6_EXTRA="--dur-cost r6d --dur-gap-qa-cost on" REMIX_ARGS="--src-duck-db -6 --unit-level on" bash scripts/r6/r6_audio.sh r6v4_14b
echo CHAIN5_DONE
