#!/bin/bash
# usage: bash inputs_r5.sh HOST — test videos + canonical + common (r3 script), briefs, references (v4, r4_video_v1, m2, m3)
H=$1; cd "$(dirname "$0")"
ssh -o BatchMode=yes $H 'mkdir -p /workspace/dub/briefs /workspace/dub/reference/{v4,r4_video_v1,m2,m3} /workspace/dub/handoff/in /workspace/dub/handoff/out /workspace/dub/outbox'
cp brief_r5_next.md /tmp/NEXT_BRIEF.md 2>/dev/null
scp -q -o BatchMode=yes /tmp/NEXT_BRIEF.md brief_r5_tracks.md user_feedback_v5_2026-10-20.md r4_audio_v1_known_issues.md pb1_listen.md r4_merged_m3_WATCH.md quality_tasks_from_spec.md LICENSES_audio.md $H:/workspace/dub/briefs/
bash inputs_r3.sh "$H"
R=/c/Users/irina/Videos/DabAI_AB
for id in 01 02 03 04 05; do
  scp -q -o BatchMode=yes $R/r3/D__${id}_r3final_fixed_v4.mp4 $H:/workspace/dub/reference/v4/${id}_v4.mp4
  scp -q -o BatchMode=yes $R/r4/V__${id}_r4_video_v1.mp4 $H:/workspace/dub/reference/r4_video_v1/${id}_r4_video_v1.mp4
  scp -q -o BatchMode=yes $R/r4/A__${id}_r4_merged_m2_WATCH.mp4 $H:/workspace/dub/reference/m2/${id}_m2.mp4 2>/dev/null
  scp -q -o BatchMode=yes $R/r4/A__${id}_r4_merged_m3_WATCH.mp4 $H:/workspace/dub/reference/m3/${id}_m3.mp4 2>/dev/null
done
ssh -o BatchMode=yes $H 'ls /workspace/dub/briefs; du -sh /workspace/dub/reference/*' 2>&1 | grep -vE "^(Welcome|Have fun|AI agents)"
echo "INPUTS_R5_DONE $H"
