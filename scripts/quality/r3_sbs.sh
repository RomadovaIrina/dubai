#!/bin/bash
# Round 3 (track A): full-video side-by-sides for the user.  r3_sbs.sh <id> <t0> <t1> <name> LABEL=run_tag ...
#   panels: ORIGINAL (25 fps master) + each LABEL=<artifacts/r3/runs/<id>_<tag>.mp4>; audio = the LAST panel's (its final mix)
#   -> outbox/<id>_<t0>-<t1>s_<name>.mp4, _half (0.5x), _zoom2x_half (2x mouth / nose crop at half speed)
set -eu
cd /workspace/dub/dubai
id=$1; t0=$2; t1=$3; name=$4; shift 4
R=/workspace/dub/artifacts/r3; M=$R/masters/$id; OUT=/workspace/dub/outbox; dur=$(python3 -c "print(round($t1-$t0,2))")
P=(--panel ORIGINAL=$M/master_25fps.mp4@0); n=1
for kv in "$@"; do P+=(--panel "${kv%%=*}=$R/runs/${id}_${kv#*=}.mp4@0"); n=$((n+1)); done
Z=$(/venv/dabai/bin/python -c "
import numpy as np; lm=np.load('$M/master_mediapipe.npz')['lm'][int($t0*25):int($t1*25)]; L=np.nanmedian(lm,0); iod=np.linalg.norm(L[33]-L[263])
cx,cy=L[[4,13]].mean(0); w=1.4*iod; h=1.4*iod
print(f'{int(cx-w/2)},{int(cy-h/2)},{int(cx+w/2)},{int(cy+h/2)}')")
b=$OUT/${id}_${t0}-${t1}s_$name
PY="scripts/nothp /venv/dabai/bin/python"
$PY scripts/quality/make_sbs.py --out $b.mp4 --start $t0 --dur $dur "${P[@]}" --audio-from $((n-1)) --title "$id ${t0}-${t1}s 25fps, audio = track C audio-r2 (final mix)"
$PY scripts/quality/make_sbs.py --out ${b}_zoom2x.mp4 --start $t0 --dur $dur "${P[@]}" --audio-from $((n-1)) --zoom $Z --title "$id mouth 2x"
for v in $b.mp4 ${b}_zoom2x.mp4; do scripts/nothp ffmpeg -loglevel error -y -nostdin -i $v -filter:v "setpts=2.0*PTS" -filter:a "atempo=0.5" -c:v libx264 -crf 18 ${v%.mp4}_half.mp4; done
rm -f ${b}_zoom2x.mp4; ls -la $b*
