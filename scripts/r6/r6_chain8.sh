#!/bin/bash
# r6 audio queue 8: pb1 reproduced a second time (pb1b = the gate's run-to-run noise floor), then the Qwen 7B arm on the v6 config; after v6
cd /workspace/dub/dubai
until grep -q "CHAIN7_DONE" /workspace/dub/artifacts/r6_chain7.log; do sleep 10; done
PAR=2 R5_EXTRA="" bash scripts/r5/r5_audio.sh pb1b
R6_QWEN=7b R6_EXTRA="--dur-cost r6c --dur-gap-qa-cost on --close-inside-pauses on" REMIX_ARGS="--src-duck-db -6 --unit-level on" bash scripts/r6/r6_audio.sh r6v6_7b
echo CHAIN8_DONE
