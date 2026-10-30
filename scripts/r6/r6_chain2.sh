#!/bin/bash
# r6 audio queue 2: v2 = v1 + --dur-cost r6c + --dur-gap-qa-cost on, 14B then 7B (replaces chain 1's v1 7B arm)
cd /workspace/dub/dubai
until grep -q "R5_AUDIO_DONE r6v1_14b" /workspace/dub/artifacts/r6_chain1.log; do sleep 10; done
kill 20349
27876 2>/dev/null; sleep 2; for p in $(pgrep -f "r4a/r6v1_7b"); do kill $p; done; sleep 3; rm -rf /workspace/dub/artifacts/r4a/r6v1_7b
R6_QWEN=14b R6_EXTRA="--dur-cost r6c --dur-gap-qa-cost on" bash scripts/r6/r6_audio.sh r6v2_14b
R6_QWEN=7b R6_EXTRA="--dur-cost r6c --dur-gap-qa-cost on" bash scripts/r6/r6_audio.sh r6v2_7b
echo CHAIN2_DONE
