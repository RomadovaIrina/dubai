#!/bin/bash
# Round 3 (track A): whole-video 25 fps video stage on the canonical cand5 work dirs.  r3_run.sh <tag> <id...> -- <extra candidate_video_stage flags>
# Base = round-2 best (r2best): clean stack, mouth-only composite, flow anchors, sub-pixel + fp32 AlignRestore, head-locked mask.
# R3_MAN_DIR / R3_DUB: dir of <id>.manifest.json / dubbed dir root (track C audio). R3_BASE=none drops the r2best flags (clean0-style arm). Shared MediaPipe / rigid caches live in artifacts/r3/masters/<id>.
set -u
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai
tag=$1; shift; ids=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do ids+=($1); shift; done; [ $# -gt 0 ] && shift
O=${R3_OUT:-/workspace/dub/artifacts/r3/runs}; mkdir -p $O
R2BEST="--anchor-filter flow:4 --anchor-round off --affine-precision fp32 --mask-track head"
[ "${R3_BASE:-r2best}" = none ] && R2BEST=""
for id in "${ids[@]}"; do
  [ -f $O/${id}_$tag.manifest.json ] && { echo "skip $id $tag"; continue; }
  M=/workspace/dub/artifacts/r3/masters/$id; W=$O/work_${id}_$tag; mkdir -p $W
  for f in master_mediapipe.npz master_mediapipe_probe2.npz master_mediapipe_2faces.npz master_rigid.npz; do [ -f $M/$f ] && ln -sfn $M/$f $W/$f; done
  ln -sfn $M/master_25fps.mp4 $W/master_25fps.mp4
  DUB_CPUS=${DUB_CPUS:-0-7} scripts/nothp ${R3_PY:-/venv/dabai/bin/python} ${R3_SCRIPT:-scripts/quality/candidate_video_stage.py} --id $id --tag $tag --manifest $( [ -n "${R3_MAN_DIR:-}" ] && echo $R3_MAN_DIR/$id.manifest.json || echo /workspace/dub/canonical/artifacts/baseline/$id/final/${id}_final.manifest.json ) ${R3_DUB:+--dubbed-dir $R3_DUB/$id} \
    --out $O --work-out $W --mux final_mix --gate-merge-gap-s 1.2 --mouth-only-pasteback on --mask-smooth 3 --ls-landmark-smooth 2 \
    --codeformer-scope mouth --codeformer off --face-stack mediapipe $R2BEST "$@" > $O/${id}_$tag.log 2>&1 || echo "FAIL $id $tag"
  for f in master_mediapipe.npz master_mediapipe_probe2.npz master_mediapipe_2faces.npz master_rigid.npz; do [ -f $W/$f ] && [ ! -L $W/$f ] && { mv $W/$f $M/$f; ln -sfn $M/$f $W/$f; }; done
  grep "^== " $O/${id}_$tag.log
done
