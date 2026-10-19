#!/bin/bash
# Round 3 track B: review clips for the user on every shaky window of <id>:
#   <id>_<t0>-<t1>s_B_ORIG-CLEAN0-R2-R3_full[_half].mp4   ORIGINAL | CLEAN0 25 fps | R2 (round-2 FIX30) | R3, all shown at 30 fps
#   <id>_<t0>-<t1>s_B_crop2x-headtracked_ORIG-R2-R3[_half].mp4   2x nose / mouth crop that follows the SOURCE head (head motion cancelled)
#   r3_clips.sh <id> <r3 tag> [r2 tag=T3] [clean0 tag=clean0] [label]
set -u
id=$1; r3=$2; r2=${3:-T3}; c0=${4:-clean0}; lab=${5:-R3}; cd /workspace/dub/dubai; R=/workspace/dub/artifacts/r3-face; O=/workspace/dub/outbox; mkdir -p $R/sbs $O
SRC=$(ls /workspace/dub/inputs/test_videos_raw/$id.*); PY=/venv/dabai/bin/python
FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0 $SRC | python3 -c "import sys;n,d=sys.stdin.read().split('/');print(round(float(n)/float(d),3))")
for w in $($PY -c "
import json; d=json.load(open('/workspace/dub/common/shaky_windows.json'))['videos'].get('$id',{}).get('windows_s',[])
for a,b in d: print(f'{a}:{b}')"); do
  t0=${w%:*}; t1=${w#*:}; dur=$(python3 -c "print(round($t1-$t0,2))"); b=${id}_${t0}-${t1}s_B
  SBS_FPS=$FPS bash scripts/quality/r2_sbs.sh $R/sbs/${b}_ORIG-CLEAN0-R2-$lab $t0 $dur none ORIGINAL=$SRC "CLEAN0 25fps=$R/gen/${id}_$c0.mp4" "R2 FIX30=$R/comp/${id}_$r2.mp4" "$lab=$R/comp/${id}_$r3.mp4" > /dev/null 2>&1
  K=$($PY -c "
import numpy as np; z=np.load('$R/timing/${id}_source_lm.npz'); t=z['t']; k=np.flatnonzero((t>=$t0)&(t<$t1)); print(f'{k.min()}:{k.max()+1}')")
  for h in "" --half; do
    $PY scripts/quality/r2_sbs_track.py --lm $R/timing/${id}_source_lm.npz --frames $K --fps $FPS --out $R/sbs/${b}_crop2x-headtracked_ORIG-R2-$lab${h:+_half}.mp4 --audio $R/comp/${id}_$r3.mp4 \
      --panel ORIGINAL=$SRC --panel "R2 FIX30=$R/comp/${id}_$r2.mp4" --panel "$lab=$R/comp/${id}_$r3.mp4" $h > /dev/null 2>&1
  done
  cp $R/sbs/${b}_*$lab*.mp4 $O/
done
ls $O | grep "^${id}_" 
