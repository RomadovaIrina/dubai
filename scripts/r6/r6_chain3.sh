#!/bin/bash
# r6 audio queue 3: v3 = v2 + --dur-cost r6d (inside-phrase pauses of the take) + --fill-slowdown 0.9 (05: a slow speaker, 0.95 left 2.4 s more source speech without dub), 14B (replaces chain 2's 7B arm)
cd /workspace/dub/dubai
until grep -q "R5_AUDIO_DONE r6v2_14b" /workspace/dub/artifacts/r6_chain2.log; do sleep 10; done
kill 27880 2>/dev/null; sleep 2; for p in $(pgrep -f "r4a/r6v2_7b"); do kill $p; done; sleep 3; rm -rf /workspace/dub/artifacts/r4a/r6v2_7b
R6_QWEN=14b R6_EXTRA="--dur-cost r6d --dur-gap-qa-cost on --fill-slowdown 0.9" bash scripts/r6/r6_audio.sh r6v3_14b
echo CHAIN3_DONE
