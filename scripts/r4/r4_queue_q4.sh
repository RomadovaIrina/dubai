#!/bin/bash
# Round 4 V queue 4: generator (+ raw layer) on audio-r4a-v1 for the first merged candidate, behind the 30 fps arm's renders. The composite
# here is the v4 stack; the ablation winner is re-composited later from the same raw layer (r4_ablate.sh-style, CPU only).
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
B=/workspace/dub/artifacts/r4v; A=$B/audio-v1
until [ -f $B/ls30/02_ls30.manifest.json ]; do sleep 30; done
for id in 04 03 05 01 02; do bash scripts/r4/r4_video.sh $id $A/$id $A/$id/${id}_r4b.manifest.json $B/m1 m1v4; done
echo Q4_DONE
