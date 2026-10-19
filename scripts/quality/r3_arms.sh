#!/bin/bash
# Round 3 (track A) window arms on top of r2best geometry: r3_arms.sh <id> <t0> <t1> <wtag> <arm>...
# Every arm = r2best (flow anchors, sub-pixel + fp32 AlignRestore, head-locked mask) + the arm's generator change.
set -u
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai
id=$1; t0=$2; t1=$3; w=$4; shift 4
O=/workspace/dub/artifacts/r3/win; M=/workspace/dub/artifacts/r3/masters/$id; mkdir -p $O
MAN=/workspace/dub/canonical/artifacts/baseline/$id/final/${id}_final.manifest.json; WB=/workspace/dub/artifacts/baseline/$id/work/work_$id
for f in master_mediapipe.npz master_rigid.npz; do [ -f $M/$f ] && ln -sfn $M/$f $WB/$f; done
f0=$(python3 -c "print(int(round($t0*25)))"); f1=$(python3 -c "print(int(round($t1*25)))")
G="--window-batch-size 1 --anchor-filter flow:4 --anchor-round off --affine-precision fp32 --mask-track head --rigid $M/master_rigid.npz --dump-stages"
for arm in "$@"; do
  case $arm in
    base) X="";;
    g2) X="--guidance 2.0";;
    g25) X="--guidance 2.5";;
    g3) X="--guidance 3.0";;
    s30) X="--steps 30";;
    nodc) X="--no-deepcache";;
    refc) X="--ref-mode fixed --ref-select calm_closed";;
    refa) X="--ref-mode fixed --ref-select calm_any";;
    refcg2) X="--ref-mode fixed --ref-select calm_closed --guidance 2.0";;
    br8k4) X="--bridge-len 8 --bridge-steps 4";;
    br16k4) X="--bridge-len 16 --bridge-steps 4";;
    br16k8) X="--bridge-len 16 --bridge-steps 8";;
    src) X="--audio src";;
    sm03) X="--silence-mix 0.3";;
    sm05) X="--silence-mix 0.5";;
    sm07) X="--silence-mix 0.7";;
    sm10) X="--silence-mix 1.0";;
    refcsm05) X="--ref-mode fixed --ref-select calm_closed --silence-mix 0.5";;
    *) echo "unknown arm $arm"; continue;;
  esac
  tag=${w}_$arm
  [ -f $O/${id}_$tag.manifest.json ] || scripts/nothp /venv/dabai/bin/python scripts/quality/ls_window_ab.py --manifest $MAN --t0 $t0 --t1 $t1 --tag $tag --out $O $G $X > $O/${id}_$tag.log 2>&1 || echo "FAIL $id $tag"
  grep "^== " $O/${id}_$tag.log
done
