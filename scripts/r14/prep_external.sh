#!/bin/bash
# r14 student data: external clips -> normalised inputs + "orig audio" stage dirs (mk_orig.py), then capture of their LatentSync windows
# through the production generator (capture_gen.py inside r7_e2e.sh, as the r13 pilot). Studio NASA interviews (face >= ~150 px at 720p;
# ISS / group shots dropped by eye: faces too small) and GRID s1 (one speaker, 360x288, ~130 px face).
#   bash scripts/r14/prep_external.sh [prep|capture|all]
set -u
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
X=/workspace/dub/r14d; DATA=$X/data; IN=$X/ext_in; AU=$X/ext_audio; CAP=$X/cap_ext; T=${SEG_T:-75}
STUDIO="n01 n02 n03 n05 c01 c02 c03 c04 c05 c06 c07 c08 c09 c10 c11 c12 c13 c14"
dur(){ ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" | cut -d. -f1; }
if [ "${1:-all}" != capture ]; then
  for v in $STUDIO; do
    f=$DATA/nasa/$v.mp4; [ -s "$f" ] || { echo "skip $v (missing)"; continue; }
    d=$(dur $f); ss=20; [ "$d" -gt 900 ] && ss=$(( d / 2 ))   # long live events open with a slate: take the middle
    python scripts/r14/mk_orig.py --video $f --id x_$v --inputs $IN --audio-root $AU --ss $ss --t $T --source "NASA $v" || echo "FAIL prep $v"
  done
  ls $DATA/grid/s1x/s1/*.mpg | head -${GRID_N:-25} > /tmp/grid_s1.txt
  python scripts/r14/mk_orig.py --concat $(cat /tmp/grid_s1.txt) --id x_grid_s1 --inputs $IN --audio-root $AU --source "GRID s1" || echo "FAIL prep grid"
fi
if [ "${1:-all}" != prep ]; then
  mkdir -p $CAP
  for d in $AU/x_*; do
    id=$(basename $d); [ -n "$(ls $CAP/${id}_*.pt 2>/dev/null | head -1)" ] && { echo "have $id"; continue; }
    t0=$(date +%s)
    DUB_ALLOW_NC=1 R13_CAPTURE_DIR=$CAP DUB_INPUTS=$IN R10_GEN_PY=/workspace/dub/dubai/scripts/r13/distill/capture_gen.py R10_JOB=$id AUDIO=$AU TAG=cap_$id SKIP_SCORE=1 \
      VIDEO_EXTRA="--codeformer off" bash scripts/r7/r7_e2e.sh $id > $X/cap_$id.log 2>&1
    echo "$id $(ls $CAP/${id}_*.pt 2>/dev/null | wc -l) windows $(( $(date +%s) - t0 )) s"
  done
  echo "CAPTURE_DONE $(ls $CAP | wc -l) windows"
fi
