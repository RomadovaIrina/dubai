#!/bin/bash
# Round 6 AUDIO: ONE command, all five videos, the same flags for every video (= r5_audio.sh with the r6 defaults).
#   r6 = pb1 + duration-aware translation (Qwen 14B) + the fixes of the r5v4 gate failures (reports/experiments/r6-audio/STATUS_LOG.md):
#     --dur-cost r6c              the dry run charges what the gate counts (dub over silence x1.0, every second outside 0.95-1.05 at full weight)
#                                 on the parts exactly as the pipeline cuts them (cut_parts pre/post-roll)
#     --dur-gap-qa-cost on        a gap-QA re-seed may not undo the duration choice (<= +0.3 s dry-run cost)
#     --fill-slowdown 0.95        no drawl: candidates / Chatterbox pace fill the source speech with words instead of atempo 0.875
#     --unit-gap 0.15             two sentences never run together (02 u6 -> u7 WER)
#     --close-inside-pauses on    Chatterbox hesitations inside an English phrase shortened to 50 ms (silent run 30 dB below the take's p90)
#     remix --unit-level on       sentence-to-sentence dub loudness (70 % to the median unit level, +-4 dB)
#     remix --gap-bed-cap-db 10   bed cap where the source speaks and the dub is silent (leaked syllables in exposed gaps);
#       --gap-bed-cap-scope edges (v2): also 150 ms into the dub edges (02 12.1 s: the leak overlapped the dub's tail)
#   = r6_audio_v2 (v1 = tag r6v8_14b; v2 = its takes + the edges cap, tag r6v9E). R6_QWEN=14b|7b (default 14b); R6_EXTRA = more flags (appended, they win)
#     bash scripts/r6/r6_audio.sh <tag> [ids]
set -u
DUB_ROOT=${DUB_ROOT:-/workspace/dub/dubai}; export DUB_ROOT; cd $DUB_ROOT; TAG=$1; shift
export DUB_CLOSE_INSIDE_DB=${DUB_CLOSE_INSIDE_DB:-30} REMIX_ARGS="${REMIX_ARGS:---src-duck-db -6 --unit-level on --gap-bed-cap-db 10 --gap-bed-cap-scope edges}"
export PAR=${PAR:-1} R5_EXTRA="--dur-translate on --qwen-size ${R6_QWEN:-14b} --dur-cost r6c --dur-gap-qa-cost on --fill-slowdown 0.95 --unit-gap 0.15 --close-inside-pauses on ${R6_EXTRA:-}"
exec bash scripts/r5/r5_audio.sh $TAG "$@"
