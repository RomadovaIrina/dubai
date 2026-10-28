#!/bin/bash
# Round 7 AUDIO: ONE command, all five videos, the same flags for every video = r6_audio_v2 (scripts/r6/r6_audio.sh: takes, tempo, pauses, bed) +
#   LIP_PLACE=on   dub blocks moved (no stretch, no new pause) into the source speech spans where the face is visible (scripts/r7/lip_place.py)
#   --gap-clean on bed in dub gaps: residual-voice / burst detector -> room tone, denoise where the source speaks (stationary over a room tone,
#                  else non-stationary 0.7: 05 residue -22.5 -> -29 dB) (scripts/r7/gap_clean.py)
#   On the published r6_audio_v1 takes (already unit-levelled): REUSE=r6v1src REMIX_ARGS="... --unit-level off ..." (README_r7.md)
#   v2: --dub-gate on (the take's noisy pauses >= 0.6 s -> -30 dB, sin^2 20-ms ramps: the 04 crackle), lip limits 2.0 / 1.5 / 0.5 s, 20-ms equal-power fades
#   then the gap metric + listen clips (scripts/r7/gap_metrics.py, vs R7_PREV when set).
#     bash scripts/r7/r7_audio.sh <tag> [ids]       REUSE=<tag> = take the TTS of an earlier r6/r7 run (no Qwen / Chatterbox); R6_QWEN=14b|7b
set -u
DUB_ROOT=${DUB_ROOT:-/workspace/dub/dubai}; export DUB_ROOT; cd $DUB_ROOT; TAG=$1; shift; IDS=${*:-01 02 03 04 05}
export LIP_PLACE=${LIP_PLACE:-on} LIP_ARGS="${LIP_ARGS:---max-shift 2.0 --grow-unit 1.5 --grow 0.5}" REMIX_ARGS="${REMIX_ARGS:---src-duck-db -6 --unit-level on --gap-bed-cap-db 10 --gap-bed-cap-scope edges --gap-clean on --gap-clean-nr-ns 0.7 --dub-gate on}"
s=$(date +%s); bash scripts/r6/r6_audio.sh $TAG $IDS
. scripts/env.sh >/dev/null; A=/workspace/dub/artifacts/r4a
python scripts/r7/gap_metrics.py --tag $TAG ${R7_PREV:+--prev $R7_PREV} --clips $A/$TAG/gaps $IDS > $A/$TAG/runs/gaps.log 2>&1 || echo "WARN gaps"
echo "R7_AUDIO_DONE $TAG $(( $(date +%s) - s )) s"
