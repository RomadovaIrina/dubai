#!/bin/bash
# usage (on laptop): bash inputs_r3.sh HOST  — test videos (Drive), canonical cand5 work dirs + common metrics (from laptop backup)
H=$1; cd "$(dirname "$0")"
DL='mkdir -p /workspace/dub/inputs /workspace/dub/handoff/in /workspace/dub/handoff/out /workspace/dub/outbox /workspace/dub/common && cd /workspace/dub/inputs && for t in 1 2 3; do [ -d test_videos_raw ] && [ $(ls test_videos_raw | wc -l) -ge 5 ] && break; rm -rf test_videos_raw 05.zip; curl -sL -m 900 -o 05.zip "https://drive.usercontent.google.com/download?id=1vRwRXh3sWnqKUQGsPJcLvDOrJ21lGfLx&export=download&confirm=t" && unzip -o -q 05.zip -d test_videos_raw; done; f=$(find test_videos_raw -type f | head -1); d=$(dirname "$f"); [ "$d" != test_videos_raw ] && mv "$d"/* test_videos_raw/ 2>/dev/null; (cd test_videos_raw && sha256sum * > ../test_videos.sha256); rm -f 05.zip; ls -la test_videos_raw; cat test_videos.sha256'
ssh -o BatchMode=yes $H "$DL" 2>&1 | grep -vE "^(Welcome|Have fun|AI agents)"
for f in /c/Users/irina/Documents/DabAI_AB_backup/B_nonvideo/artifacts_baseline_0*_.tar; do
  for t in 1 2 3 4; do ssh -o BatchMode=yes -o ServerAliveInterval=15 $H 'mkdir -p /workspace/dub/canonical && tar xf - -C /workspace/dub/canonical' < $f 2>/dev/null && break; done
done
tar cf - -C common mouth_jitter_hr.py slide_metric.py shaky_windows.json $( [ -f common/anchor_filters.py ] && echo anchor_filters.py) | ssh -o BatchMode=yes $H 'tar xf - -C /workspace/dub/common'
ssh -o BatchMode=yes $H 'du -sh /workspace/dub/canonical/artifacts/baseline/* /workspace/dub/common/*' 2>&1 | grep -vE "^(Welcome|Have fun|AI agents)"
echo "INPUTS_DONE $H"
