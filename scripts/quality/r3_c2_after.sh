#!/bin/bash
# r3 B: after each C-audio render (gen/<id>_c2F) -> R3 composite (comp/<id>_c2_R3), SyncNet, whole-video deliverables in outbox/
cd /workspace/dub/dubai; R=/workspace/dub/artifacts/r3-face; S=/workspace/dub/inputs/test_videos_raw; O=/workspace/dub/outbox
X="--colour-match upper --seam-lf 4 --sharpen-auto 1.0 --skin-texture source"
for id in "$@"; do
  until grep -q "^== $id c2F\|^FAIL $id c2F" $R/gen/${id}_c2F.log 2>/dev/null || grep -q "FAIL $id c2F" $R/gpuq/logs/*.log 2>/dev/null; do sleep 20; done
  GEN=c2F scripts/quality/r3_compF.sh $id c2_R3 $X
  SRC=$(ls $S/$id.*); dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $SRC); FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0 $SRC | python3 -c "import sys;n,d=sys.stdin.read().split('/');print(round(float(n)/float(d),3))")
  cp $R/comp/${id}_c2_R3.mp4 $O/${id}_full_R3face_Caudio-r2.mp4
  SBS_AUDIO=2 SBS_FPS=$FPS bash scripts/quality/r2_sbs.sh $R/sbs/${id}_whole_ORIG-CLEAN0-R3Caudio 0 $dur none ORIGINAL=$SRC "CLEAN0 25fps (cand5 audio)=$R/gen/${id}_clean0.mp4" "R3 face + C audio-r2=$R/comp/${id}_c2_R3.mp4" > /dev/null 2>&1
  cp $R/sbs/${id}_whole_ORIG-CLEAN0-R3Caudio_full.mp4 $O/${id}_whole_B_ORIG-CLEAN0-R3Caudio_sbs.mp4
  echo "$id done $(date +%H:%M)"
done
