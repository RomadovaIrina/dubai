#!/bin/bash
# Round 3 D: video stage (25 fps generator) on the canonical cand5 audio (or a delivered audio dir) for one arm.
#   bash scripts/quality/r3_video.sh <arm> <id...>     arms: clean0 | r2m (A r2best generator flags + raw LS dump for FIX30)
#   DUBBED_DIR_TPL=/path/{id}  -> use that dubbed dir (dubbed_16k.wav + final_mix_aac.m4a) instead of the canonical work dir
set -u
DUB_ROOT=/workspace/dub/dubai; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
arm=$1; shift; O=${R3D_OUT:-/workspace/dub/artifacts/r3d}/video; mkdir -p $O
CLEAN0="--mux final_mix --gate-merge-gap-s 1.2 --mouth-only-pasteback on --mask-smooth 3 --ls-landmark-smooth 2 --codeformer-scope mouth --codeformer off --face-stack mediapipe"
case $arm in
  clean0) X="";;
  r2m) X="--anchor-filter flow:4 --anchor-round off --affine-precision fp32 --mask-track head --dump-ls-raw on";;
  *) X=${ARM_FLAGS:?unknown arm: set ARM_FLAGS};;
esac
for id in "$@"; do
  C=/workspace/dub/canonical/artifacts/baseline/$id; D=/workspace/dub/artifacts/baseline/$id/work/work_$id; mkdir -p $D
  for f in $C/work/work_$id/*; do ln -sfn $f $D/; done
  DD=""; [ -n "${DUBBED_DIR_TPL:-}" ] && DD="--dubbed-dir ${DUBBED_DIR_TPL//\{id\}/$id}"
  MAN=$C/final/${id}_final.manifest.json; [ -n "${MANIFEST_TPL:-}" ] && MAN=${MANIFEST_TPL//\{id\}/$id}   # e.g. C's delivered <id>_r2.manifest.json (its units / text)
  s=$(date +%s)
  flock /workspace/dub/gpu.lock python scripts/quality/candidate_video_stage.py --id $id --tag $arm --manifest $MAN --out $O --work-out $O/work_${id}_$arm \
     $DD $CLEAN0 $X > $O/${id}_$arm.log 2>&1 || echo "FAIL $id $arm (see $O/${id}_$arm.log)"
  echo "$(grep '^== ' $O/${id}_$arm.log | cut -c1-200)  wall $(( $(date +%s) - s )) s"
done
