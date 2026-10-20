#!/bin/bash
# usage: bash inputs_r4.sh HOST — r3 inputs (test videos, canonical, common) + v4 reference finals + round-4 briefs
H=$1; cd "$(dirname "$0")"
bash inputs_r3.sh "$H"
ssh -o BatchMode=yes $H 'mkdir -p /workspace/dub/reference/v4 /workspace/dub/briefs /workspace/dub/handoff/in /workspace/dub/handoff/out /workspace/dub/outbox'
for id in 01 02 03 04 05; do for t in 1 2 3; do scp -q -o BatchMode=yes /c/Users/irina/Videos/DabAI_AB/r3/D__${id}_r3final_fixed_v4.mp4 $H:/workspace/dub/reference/v4/${id}_v4.mp4 && break; done; done
scp -q -o BatchMode=yes brief_r4_common.md brief_r4_audio.md brief_r4_video.md user_feedback_v5_2026-10-20.md quality_tasks_from_spec.md LICENSES_audio.md $H:/workspace/dub/briefs/
ssh -o BatchMode=yes $H 'ls -la /workspace/dub/reference/v4 /workspace/dub/briefs; sha256sum /workspace/dub/reference/v4/*' 2>&1 | grep -vE "^(Welcome|Have fun|AI agents)"
echo "INPUTS_R4_DONE $H"
