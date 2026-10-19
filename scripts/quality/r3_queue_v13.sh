#!/bin/bash
# r3 D queue v13: 05 r3final render (re-run after the OOM) -> v3 composites 05 / 01 / 02 -> publish
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; V=/workspace/dub/artifacts/r3d/video; A4=/workspace/dub/artifacts/r3d/audio-r4
while pgrep -f "r3_video[.]sh r3(nb|g15) 04" >/dev/null; do sleep 20; done
[ -f $V/05_r3final.manifest.json ] || DUBBED_DIR_TPL=$A4/05 MANIFEST_TPL=$A4/05/05_r4.manifest.json bash scripts/quality/r3_video.sh r3final 05
for id in 05 01 02; do FIX30_MODE=r3fix FIX30_THREADS=12 bash scripts/quality/r3_fix30.sh $V/${id}_r3final.manifest.json v3; done
bash scripts/quality/r3_publish_v3.sh 05 01 02; echo V13_DONE
