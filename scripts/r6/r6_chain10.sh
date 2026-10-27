#!/bin/bash
# r6 audio queue 10: v8 = v7 + close-inside threshold 30 dB, 14B; after chain 9 (v7 7B)
cd /workspace/dub/dubai
until grep -q "CHAIN9_DONE" /workspace/dub/artifacts/r6_chain9.log; do sleep 10; done
export DUB_CLOSE_INSIDE_DB=30 REMIX_ARGS="--src-duck-db -6 --unit-level on"
R6_QWEN=14b R6_EXTRA="--dur-cost r6c --dur-gap-qa-cost on --close-inside-pauses on" bash scripts/r6/r6_audio.sh r6v8_14b
echo CHAIN10_DONE
