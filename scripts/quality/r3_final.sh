#!/bin/bash
# r3 B: FINAL R3 composite on C's audio-r2 renders (gen/<id>_c2F) -> comp/<id>_c2_R3f + eval + SyncNet + outbox full video.
cd /workspace/dub/dubai; R=/workspace/dub/artifacts/r3-face; O=/workspace/dub/outbox
R3FLAGS="--colour-match upper --seam-lf 4 --sharpen-auto 0.95 --skin-texture source --lip-guard on"
for id in "$@"; do
  GEN=c2F scripts/quality/r3_compF.sh $id c2_R3f $R3FLAGS
  scripts/quality/r3_sync.sh $id:c2_R3f
  cp $R/comp/${id}_c2_R3f.mp4 $O/${id}_full_R3face_audio-C-r2.mp4
done
