#!/bin/bash
# r6 audio queue 9: after pb1b, v7 = v6 + close-inside-pauses on the silent run itself (3da8b24), 14B then 7B (the Qwen choice)
cd /workspace/dub/dubai
until grep -q "R5_AUDIO_DONE pb1b" /workspace/dub/artifacts/r6_chain8.log; do sleep 10; done
X="--dur-cost r6c --dur-gap-qa-cost on --close-inside-pauses on"; export REMIX_ARGS="--src-duck-db -6 --unit-level on"
R6_QWEN=14b R6_EXTRA="$X" bash scripts/r6/r6_audio.sh r6v7_14b
R6_QWEN=7b R6_EXTRA="$X" bash scripts/r6/r6_audio.sh r6v7_7b
echo CHAIN9_DONE
