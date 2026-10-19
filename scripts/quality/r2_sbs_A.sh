#!/bin/bash
# Round 2 (track A): side-by-side clips ORIGINAL | CLEAN0 | variants at normal + half speed, and a 2x nose / mouth zoom (normal + half speed).
#   r2_sbs_A.sh <id> <t0> <t1> <wtag> <arm>...      -> /workspace/dub/artifacts/lipsync-temporal/sbs/r2/<id>_<wtag>_*.mp4
set -eu
cd /workspace/dub/dubai
id=$1; t0=$2; t1=$3; w=$4; shift 4
O=/workspace/dub/artifacts/lipsync-temporal/r2/win; M=/workspace/dub/artifacts/lipsync-temporal/r2/masters/$id; S=/workspace/dub/artifacts/lipsync-temporal/sbs/r2
mkdir -p $S; dur=$(python3 -c "print(round($t1-$t0,2))")
P=(--panel ORIGINAL=$M/master_25fps.mp4@0)
for arm in "$@"; do P+=(--panel "$(echo $arm | tr a-z A-Z)=$O/${id}_${w}_$arm.mp4@$t0"); done
# nose / mouth zoom box: median landmarks over the window (nose bridge to chin, cheek to cheek)
Z=$(/venv/dabai/bin/python -c "
import numpy as np; lm=np.load('$M/master_mediapipe.npz')['lm'][int($t0*25):int($t1*25)]; L=np.nanmedian(lm,0); iod=np.linalg.norm(L[33]-L[263])
cx,cy=L[[4,13]].mean(0); w=1.3*iod; h=1.3*iod
print(f'{int(cx-w/2)},{int(cy-h/2)},{int(cx+w/2)},{int(cy+h/2)}')")
base=$S/${id}_${w}_$(echo "$@" | tr ' ' '-')
/venv/dabai/bin/python scripts/quality/make_sbs.py --out ${base}.mp4 --start $t0 --dur $dur "${P[@]}" --audio-from 1 --title "$id $w ${t0}-${t1}s 25fps"
/venv/dabai/bin/python scripts/quality/make_sbs.py --out ${base}_zoom2x.mp4 --start $t0 --dur $dur "${P[@]}" --audio-from 1 --zoom $Z --title "$id $w nose/mouth 2x"
for v in ${base}.mp4 ${base}_zoom2x.mp4; do
  ffmpeg -loglevel error -y -nostdin -i $v -filter:v "setpts=2.0*PTS" -filter:a "atempo=0.5" -c:v libx264 -crf 18 ${v%.mp4}_half.mp4
done
ls -la ${base}*
