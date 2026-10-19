#!/bin/bash
# Round 3 track B: 25 fps video stage on the canonical cand5 work dirs + the raw LatentSync layer for fps_composite.py.
#   r3_render.sh <tag> <id...> -- <extra candidate_video_stage flags>
#   tags used: gen = A's round-2 generator (flow4nr32hm: flow anchors, sub-pixel, fp32 AlignRestore, head-locked mask); clean0 = clean-base
set -u
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai
tag=$1; shift; ids=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do ids+=($1); shift; done; [ $# -gt 0 ] && shift
O=/workspace/dub/artifacts/r3-face/gen; mkdir -p $O
for id in "${ids[@]}"; do
  C=/workspace/dub/canonical/artifacts/baseline/$id/work/work_$id; D=/workspace/dub/artifacts/baseline/$id/work/work_$id; mkdir -p $D
  for f in $C/*; do ln -sfn $f $D/; done   # the manifest's own work-dir path -> canonical audio stages
  [ -f $O/${id}_$tag.manifest.json ] && { echo "skip $id $tag"; continue; }
  /venv/dabai/bin/python scripts/quality/candidate_video_stage.py --id $id --tag $tag --manifest /workspace/dub/canonical/artifacts/baseline/$id/final/${id}_final.manifest.json \
    --out $O --work-out $O/work_${id}_$tag --mux final_mix --gate-merge-gap-s 1.2 --mouth-only-pasteback on --mask-smooth 3 --ls-landmark-smooth 2 \
    --codeformer-scope mouth --codeformer off --face-stack mediapipe --dump-ls-raw $O/${id}_${tag}_ls_raw.mp4 "$@" > $O/${id}_$tag.log 2>&1 || echo "FAIL $id $tag"
  grep "^== \|\[dump\]" $O/${id}_$tag.log
done
