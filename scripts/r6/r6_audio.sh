#!/bin/bash
# Round 6 AUDIO: ONE command, all five videos, the same flags for every video (= r5_audio.sh with the r6 defaults).
#   r6 = pb1 + duration-aware translation + the fixes of the four r5v4 gate failures:
#     --dur-cost r6        the dry run charges what the gate counts (dub over silence x1.0, every second outside 0.95-1.05 at full weight)
#     --fill-slowdown 0.95 no drawl: candidates / Chatterbox pace fill the source speech with words instead of atempo 0.875
#     --unit-gap 0.15      two sentences never run together (02 u6 -> u7 WER)
#   R6_QWEN=14b|7b (default 14b); R6_EXTRA = more flags (appended, they win)
#     bash scripts/r6/r6_audio.sh <tag> [ids]
set -u
DUB_ROOT=${DUB_ROOT:-/workspace/dub/dubai}; export DUB_ROOT; cd $DUB_ROOT; TAG=$1; shift
export PAR=${PAR:-1} R5_EXTRA="--dur-translate on --qwen-size ${R6_QWEN:-14b} --dur-cost r6 --fill-slowdown 0.95 --unit-gap 0.15 ${R6_EXTRA:-}"
exec bash scripts/r5/r5_audio.sh $TAG "$@"
