#!/bin/bash
# Round 5 AUDIO: ONE command, all five videos, the same flags for every video.
#   audio of record pb1 = r4_audio.sh + --ne-guard on --pause-boundary-only on + bed --src-duck-db -6 (scripts/r4/r4_e2e.sh)
#   r5 vN       = pb1 + R5_EXTRA (default: --dur-translate on, duration-aware translation, scripts/pilot/dur_translate.py)
#   then the faithfulness axis (r5_faith.py, Qwen judge + back-translation) on every video.
#     bash scripts/r5/r5_audio.sh <tag> [ids]        R5_EXTRA="" -> pb1 itself (the baseline reproduced on this server)
set -u
DUB_ROOT=/workspace/dub/dubai; export DUB_ROOT; cd $DUB_ROOT; TAG=$1; shift; IDS=${*:-01 02 03 04 05}
export AUDIO_EXTRA="--ne-guard on --pause-boundary-only on ${R5_EXTRA---dur-translate on}" REMIX_ARGS="${REMIX_ARGS:---src-duck-db -6}"
s=$(date +%s); bash scripts/r4_audio.sh $TAG $IDS
. scripts/env.sh >/dev/null; export PYTHONPATH=third_party/llama_cuda${PYTHONPATH:+:$PYTHONPATH}
python scripts/r5/r5_faith.py $TAG $IDS > /workspace/dub/artifacts/r4a/$TAG/runs/faith.log 2>&1 || echo "WARN faith"
echo "R5_AUDIO_DONE $TAG $(( $(date +%s) - s )) s"
