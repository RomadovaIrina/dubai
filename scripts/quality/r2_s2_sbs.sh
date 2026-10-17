#!/bin/bash
# Round 2 stage 2: side-by-sides from the FULL 25 fps videos on the shaky windows: ORIGINAL | CLEAN0 | R2BEST (normal, half, 2x zoom, head-stabilised)
#   r2_s2_sbs.sh <id> <t0> <t1>
set -eu
cd /workspace/dub/dubai
id=$1; t0=$2; t1=$3; R=/workspace/dub/artifacts/lipsync-temporal/r2; M=$R/masters/$id; S=/workspace/dub/artifacts/lipsync-temporal/sbs/full
C=$R/s2/${id}_s2_clean0.mp4; B=$R/s2/${id}_s2_flowhm.mp4; dur=$(python3 -c "print(round($t1-$t0,2))"); b=$S/${id}_${t0}-${t1}s_orig-clean0-r2best
Z=$(/venv/dabai/bin/python -c "
import numpy as np; lm=np.load('$M/master_mediapipe.npz')['lm'][int($t0*25):int($t1*25)]; L=np.nanmedian(lm,0); iod=np.linalg.norm(L[33]-L[263]); cx,cy=L[[4,13]].mean(0); w=1.3*iod
print(f'{int(cx-w/2)},{int(cy-w/2)},{int(cx+w/2)},{int(cy+w/2)}')")
P=(--panel ORIGINAL=$M/master_25fps.mp4@0 --panel CLEAN0=$C@0 --panel R2BEST=$B@0)
/venv/dabai/bin/python scripts/quality/make_sbs.py --out $b.mp4 --start $t0 --dur $dur "${P[@]}" --audio-from 2 --title "$id ${t0}-${t1}s 25fps" > /dev/null
/venv/dabai/bin/python scripts/quality/make_sbs.py --out ${b}_zoom2x.mp4 --start $t0 --dur $dur "${P[@]}" --audio-from 2 --zoom $Z --title "$id nose/mouth 2x" > /dev/null
for v in $b.mp4 ${b}_zoom2x.mp4; do ffmpeg -loglevel error -y -nostdin -i $v -filter:v "setpts=2.0*PTS" -filter:a "atempo=0.5" -c:v libx264 -crf 18 ${v%.mp4}_half.mp4; done
/venv/dabai/bin/python scripts/quality/r2_stab_sbs.py --id $id --t0 $t0 --t1 $t1 --panel CLEAN0=$C@0 --panel R2BEST=$B@0 --diff --half --out ${b}_stab2x.mp4 > /dev/null
ls ${b}*
