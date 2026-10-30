#!/bin/bash
# r6 audio queue 4: the Qwen 7B arm on the v3 config (the 14B-vs-7B choice), after r6v3_14b
cd /workspace/dub/dubai
until grep -q "CHAIN3_DONE" /workspace/dub/artifacts/r6_chain3.log; do sleep 10; done
R6_QWEN=7b R6_EXTRA="--dur-cost r6d --dur-gap-qa-cost on --fill-slowdown 0.9" bash scripts/r6/r6_audio.sh r6v3_7b
echo CHAIN4_DONE
