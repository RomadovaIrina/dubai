#!/bin/bash
# r3 B: after each A-recipe render on C audio-r2 (gen/<id>_a3F: r2best, window batch 1, C's manifest / gate, on-time master)
#   -> final R3 composite comp/<id>_a3_R3 + eval + SyncNet -> outbox full video + window clips + whole-video SBS (all audio-r2).
cd /workspace/dub/dubai; R=/workspace/dub/artifacts/r3-face; S=${DUB_INPUTS:-/workspace/dub/inputs/test_videos_raw}; O=/workspace/dub/outbox
R3FLAGS="--colour-match upper --seam-lf 4 --sharpen-auto 0.95 --skin-texture source --lip-guard on"
for id in "$@"; do
  until grep -q "^== $id a3F\|^FAIL $id a3F" $R/gen/${id}_a3F.log 2>/dev/null; do sleep 20; done
  GEN=a3F scripts/quality/r3_compF.sh $id a3_R3 $R3FLAGS
  /venv/dabai/bin/python scripts/quality/r3_joints.py --video $R/comp/${id}_a3_R3.mp4 --manifest $R/gen/${id}_a3F.manifest.json --lm $R/timing/${id}_source_lm_F.npz --out $R/metrics/${id}_a3_R3.joints.json > /dev/null
  sed -i 's#case $tag in c2\*)#case $tag in c2*|a3*)#' scripts/quality/r3_sync.sh
  scripts/quality/r3_sync.sh $id:a3_R3 $id:a3F
  cp $R/comp/${id}_a3_R3.mp4 $O/${id}_full_R3face_audio-C-r2.mp4
  until [ -f $R/comp/${id}_c2_T3.mp4 ]; do sleep 20; done
  AUD=C-r2 scripts/quality/r3_clips.sh $id a3_R3 c2_T3 c2c0 R3
  SRC=$(ls $S/$id.*); dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $SRC); FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0 $SRC | python3 -c "import sys;n,d=sys.stdin.read().split('/');print(round(float(n)/float(d),3))")
  SBS_AUDIO=1 SBS_FPS=$FPS bash scripts/quality/r2_sbs.sh $R/sbs/${id}_whole_audio-C-r2_ORIG-CLEAN0-R3a3 0 $dur none ORIGINAL=$SRC "CLEAN0 25fps=$R/gen/${id}_c2c0.mp4" "R3=$R/comp/${id}_a3_R3.mp4" > /dev/null 2>&1
  cp $R/sbs/${id}_whole_audio-C-r2_ORIG-CLEAN0-R3a3_full.mp4 $O/${id}_whole_B_audio-C-r2_ORIG-CLEAN0-R3_sbs.mp4
  echo "$id a3 done $(date +%H:%M)"
done
