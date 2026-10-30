#!/bin/bash
# r6 audio queue 1: pb1 reproduced on this server (gate baseline), then r6v1 with Qwen 14B and with 7B (the 14B-vs-7B choice)
cd /workspace/dub/dubai
PAR=2 R5_EXTRA="" bash scripts/r5/r5_audio.sh pb1
R6_QWEN=14b bash scripts/r6/r6_audio.sh r6v1_14b
R6_QWEN=7b bash scripts/r6/r6_audio.sh r6v1_7b
echo CHAIN1_DONE
