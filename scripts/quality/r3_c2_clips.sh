#!/bin/bash
# r3 B: user clips on C's audio-r2 only: R2 panel = round-2 FIX30 (T3) on A's generator with the -r 25 master (gen/<id>_c2g),
# CLEAN0 panel = clean-base video stage (gen/<id>_c2c0), R3 = comp/<id>_c2_R3f. Window clips + whole-video side-by-side.
cd /workspace/dub/dubai; R=/workspace/dub/artifacts/r3-face; S=/workspace/dub/inputs/test_videos_raw; O=/workspace/dub/outbox
for id in "$@"; do
  until grep -q "^== $id c2g" $R/gen/${id}_c2g.log 2>/dev/null; do sleep 20; done
  [ -f $R/comp/${id}_c2_T3.mp4 ] || /venv/dabai/bin/python scripts/quality/fps_composite.py --manifest $R/gen/${id}_c2g.manifest.json --ls-raw $R/gen/${id}_c2g_ls_raw.mp4 \
     --out $R/comp/${id}_c2_T3.mp4 --master-timing $R/timing/${id}_master_timing.json --register sim_ring --threads 12 | tail -1
  AUD=C-r2 scripts/quality/r3_clips.sh $id c2_R3f c2_T3 c2c0 R3
  SRC=$(ls $S/$id.*); dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $SRC); FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0 $SRC | python3 -c "import sys;n,d=sys.stdin.read().split('/');print(round(float(n)/float(d),3))")
  SBS_AUDIO=1 SBS_FPS=$FPS bash scripts/quality/r2_sbs.sh $R/sbs/${id}_whole_audio-C-r2_ORIG-CLEAN0-R3 0 $dur none ORIGINAL=$SRC "CLEAN0 25fps=$R/gen/${id}_c2c0.mp4" "R3=$R/comp/${id}_c2_R3f.mp4" > /dev/null 2>&1
  cp $R/sbs/${id}_whole_audio-C-r2_ORIG-CLEAN0-R3_full.mp4 $O/${id}_whole_B_audio-C-r2_ORIG-CLEAN0-R3_sbs.mp4
  echo "$id clips done $(date +%H:%M)"
done
