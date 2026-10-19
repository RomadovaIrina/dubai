#!/bin/bash
# Round 3 track B: evaluation of one source-fps output: r3_flicker (whole video), common slide metric + v4 on every shaky window.
#   r3_eval.sh <id> <tag> <video.mp4> [manifest]   -> /workspace/dub/artifacts/r3-face/metrics/<id>_<tag>.*
set -u
id=$1; tag=$2; vid=$3; man=${4:-}
cd /workspace/dub/dubai; export SLIDE_DIR=/workspace/dub/artifacts/r3-face; R=$SLIDE_DIR; M=$R/metrics; mkdir -p $M
PY=/venv/dabai/bin/python; SRC=$(ls /workspace/dub/inputs/test_videos_raw/$id.*)
$PY scripts/quality/r3_flicker.py --video $vid --ref $SRC --lm $R/timing/${id}_source_lm.npz --out $M/${id}_$tag.flicker.json --series > /dev/null 2>&1 &
for w in $($PY -c "
import json,numpy as np; d=json.load(open('/workspace/dub/common/shaky_windows.json'))['videos'].get('$id',{}).get('windows_s',[])
for a,b in d: print(f'{int(round(a*25))}:{int(round(b*25))}')"); do
  bash scripts/quality/r2_slide_fps.sh $id $w $M/${id}_${tag}_slide_${w/:/_}.json $vid > /dev/null 2>&1
  K=$($PY -c "
import numpy as np; z=np.load('$R/timing/${id}_source_lm.npz'); mu=z['mu']; A,B=[int(x) for x in '$w'.split(':')]
k=np.flatnonzero((mu>=A+2)&(mu<=B-3)); print(f'{k.min()}:{k.max()+1}')")
  OPENCV_FOR_THREADS_NUM=6 $PY scripts/quality/mouth_jitter_hr.py --video $vid --reference $SRC --frames $K ${man:+--manifest $man} --out $M/${id}_${tag}_v4_${w/:/_}.json --tag $tag > /dev/null 2>&1
done
wait
$PY - <<PY
import json, glob
f = json.load(open('$M/${id}_${tag}.flicker.json')); e = f['excess']
print('$id $tag flicker: band_up %.3f skin_up %.3f band_low %.3f chin %.3f upper %.3f | colour %.3f | sharp %.3f | score %.3f / p95 %.3f' % (e['band_up']['mean'], e['skin_up']['mean'], e['band_low']['mean'], e['chin']['mean'], e['upper']['mean'], f['colour_flicker']['mean'], f['sharpness_ratio'], f['score']['mean'], f['score']['p95']))
for p in sorted(glob.glob('$M/${id}_${tag}_slide_*.json')):
    r = json.load(open(p))['regions']; print('  slide', p.split('_slide_')[1][:-5], 'nosebase abs %.3f slide %.3f p95 %.3f | cheeks slide %.3f | upper abs %.3f' % (r['nosebase']['abs_px']['mean'], r['nosebase']['slide_px']['mean'], r['nosebase']['slide_px']['p95'], r['cheeks']['slide_px']['mean'], r['upper']['abs_px']['mean']))
PY
