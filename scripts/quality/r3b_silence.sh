#!/bin/bash
# r3 B: dub-silent stretches where the source speaks (C's residual gaps): OPT1 still lower face / OPT2 original pass-through, eased,
# on top of the final R3 composite; 4-panel clips ORIGINAL | R3 | OPT1 | OPT2 (+ half speed) around every gap >= 0.6 s. audio-r2 only.
cd /workspace/dub/dubai; R=/workspace/dub/artifacts/r3-face; S=${DUB_INPUTS:-/workspace/dub/inputs/test_videos_raw}; O=/workspace/dub/outbox
R3FLAGS="--colour-match upper --seam-lf 4 --sharpen-auto 0.95 --skin-texture source --lip-guard on"
for id in "$@"; do
  until [ -f $R/comp/${id}_a3_R3.mp4 ] && [ -f $R/metrics/${id}_a3_R3.syncnet.json ]; do sleep 30; done
  for opt in still orig; do
    [ -f $R/comp/${id}_a3_R3sil_$opt.mp4 ] || GEN=a3F scripts/quality/r3_compF.sh $id a3_R3sil_$opt $R3FLAGS --silence $opt --gaps $R/gaps/${id}_residual.json > /dev/null 2>&1
  done
  SRC=$(ls $S/$id.*); FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0 $SRC | python3 -c "import sys;n,d=sys.stdin.read().split('/');print(round(float(n)/float(d),3))")
  for g in $(/venv/dabai/bin/python -c "
import json; [print(f\"{max(0,x['start']-1.5):.1f}:{x['end']+1.5:.1f}\") for x in json.load(open('$R/gaps/${id}_residual.json'))['gaps'] if x['end']-x['start']>=0.6]"); do
    t0=${g%:*}; t1=${g#*:}; dur=$(python3 -c "print(round($t1-$t0,2))")
    SBS_AUDIO=1 SBS_FPS=$FPS bash scripts/quality/r2_sbs.sh $R/sbs/${id}_${t0}-${t1}s_B_audio-C-r2_dubsilent_ORIG-R3-OPT1still-OPT2orig $t0 $dur none ORIGINAL=$SRC "R3=$R/comp/${id}_a3_R3.mp4" "OPT1 still lower face=$R/comp/${id}_a3_R3sil_still.mp4" "OPT2 original lips=$R/comp/${id}_a3_R3sil_orig.mp4" > /dev/null 2>&1
    cp $R/sbs/${id}_${t0}-${t1}s_B_audio-C-r2_dubsilent_*.mp4 $O/
  done
  echo "$id silence done $(date +%H:%M)"
done
