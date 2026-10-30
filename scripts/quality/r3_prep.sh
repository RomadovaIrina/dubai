#!/bin/bash
# Round 3 (track A): per-video shared inputs. Links the canonical cand5 work dir at the path its manifest records
# (/workspace/dub/artifacts/baseline/<id>/work/work_<id>) and builds the 25 fps master once (same ffmpeg call as make_master()).
# MediaPipe / rigid caches are filled by the first run and shared through r3_run.sh.   r3_prep.sh <id>...
set -e
for id in "$@"; do
  C=/workspace/dub/canonical/artifacts/baseline/$id/work/work_$id; D=/workspace/dub/artifacts/baseline/$id/work/work_$id; M=/workspace/dub/artifacts/r3/masters/$id
  mkdir -p $D $M
  for f in $C/*; do ln -sfn $f $D/; done
  src=$(ls ${DUB_INPUTS:-/workspace/dub/inputs/test_videos_raw}/$id.* | head -1)
  [ -f $M/master_25fps.mp4 ] || ffmpeg -loglevel error -y -nostdin -i $src -r 25 -crf 18 -an $M/master_25fps.mp4
  ln -sfn $M/master_25fps.mp4 $D/master_25fps.mp4
  echo "$id: $(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $M/master_25fps.mp4) master frames"
done
