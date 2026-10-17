#!/bin/bash
# Round 2 (track A) arm queue: r2_arms.sh <id> <t0> <t1> <wtag> <arm>...   arms: clean0 raw flow4 flow4nr flow4nr32 c0nr32 kal sub3nr32 flow2nr32 flow8nr32
set -u
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai
id=$1; t0=$2; t1=$3; w=$4; shift 4
O=/workspace/dub/artifacts/lipsync-temporal/r2/win; M=/workspace/dub/artifacts/lipsync-temporal/r2/masters; A=/workspace/dub/artifacts/lipsync-temporal/r2/anchor
MAN=/workspace/dub/canonical/artifacts/baseline/$id/final/${id}_final.manifest.json
f0=$(python3 -c "print(int(round($t0*25)))"); f1=$(python3 -c "print(int(round($t1*25)))")
for arm in "$@"; do
  case $arm in
    clean0) X="";;
    raw) X="--anchor-filter raw";;
    rawnr32) X="--anchor-filter raw --anchor-round off --affine-precision fp32";;
    flow4) X="--anchor-filter flow:4";;
    flow4nr) X="--anchor-filter flow:4 --anchor-round off";;
    flow4nr32) X="--anchor-filter flow:4 --anchor-round off --affine-precision fp32";;
    flow2nr32) X="--anchor-filter flow:2 --anchor-round off --affine-precision fp32";;
    flow8nr32) X="--anchor-filter flow:8 --anchor-round off --affine-precision fp32";;
    c032) X="--affine-precision fp32";;
    c0nr) X="--anchor-round off";;
    c0nr32) X="--anchor-round off --affine-precision fp32";;
    kal) X="--anchor-filter kalman:5.0:0.5";;
    kalnr32) X="--anchor-filter kalman:5.0:0.5 --anchor-round off --affine-precision fp32";;
    sub3nr32) X="--anchor-filter sub:3:4 --anchor-round off --affine-precision fp32";;
    wrap) X="--mode wrapper";;
    wrapnr) X="--mode wrapper --anchor-round off";;
    wrap32) X="--mode wrapper --affine-precision fp32";;
    wrapnr32) X="--mode wrapper --anchor-round off --affine-precision fp32";;
    wrapflownr32) X="--mode wrapper --anchor-filter flow:4 --anchor-round off --affine-precision fp32";;
    *) echo "unknown arm $arm"; continue;;
  esac
  tag=${w}_$arm
  if [ ! -f $O/${id}_$tag.manifest.json ]; then
    /venv/dabai/bin/python scripts/quality/ls_window_ab.py --manifest $MAN --t0 $t0 --t1 $t1 --tag $tag --out $O --rigid $A/${id}_rigid.npz $X > $O/${id}_$tag.log 2>&1 || echo "FAIL $id $tag"
  fi
  /venv/dabai/bin/python scripts/quality/slide_metric.py --video $O/${id}_$tag.mp4 --master $M/$id/master_25fps.mp4 --npz $M/$id/master_mediapipe.npz --offset $f0 --frames $f0:$f1 \
    --boxes $O/${id}_${tag}_maskboxes.npz --manifest $O/${id}_$tag.manifest.json --rigid $A/${id}_rigid.npz --series --out $O/${id}_$tag.slide.json | tail -1
  grep "^== " $O/${id}_$tag.log
done
