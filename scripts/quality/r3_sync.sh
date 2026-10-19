#!/bin/bash
# r3 B: SyncNet (mix / dub-only) of <id>:<tag> outputs (comp/<id>_<tag>.mp4, or gen/<id>_<tag>.mp4 for 25 fps renders); skips existing.
cd /workspace/dub/dubai; R=/workspace/dub/artifacts/r3-face
for it in "$@"; do id=${it%%:*}; tag=${it#*:}; v=$R/comp/${id}_$tag.mp4; [ -f $v ] || v=$R/gen/${id}_$tag.mp4
  [ -f $R/metrics/${id}_$tag.syncnet.json ] && continue
  dub=/workspace/dub/canonical/artifacts/baseline/$id/work/work_$id/dubbed_aac.m4a; case $tag in c2*|a3*) dub=/workspace/dub/handoff/in/C/audio-r2/$id/dubbed_aac.m4a;; c3*) dub=/workspace/dub/handoff/in/C/audio-r3/$id/dubbed_aac.m4a;; esac   # C audio-r2 renders
  DUB_ROOT=/workspace/dub/dubai /venv/dabai/bin/python scripts/quality/syncnet_fps.py $v --dub $dub --json $R/metrics/${id}_$tag.syncnet.json 2>/dev/null | tail -1
done
