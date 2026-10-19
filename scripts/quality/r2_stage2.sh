#!/bin/bash
# Round 2 stage 2: whole-video video stage at 25 fps on the canonical work dirs.  r2_stage2.sh <tag> <id...> -- <extra candidate_video_stage flags>
set -u
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai
tag=$1; shift; ids=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do ids+=($1); shift; done; [ $# -gt 0 ] && shift
O=/workspace/dub/artifacts/lipsync-temporal/r2/s2
for id in "${ids[@]}"; do
  [ -f $O/${id}_$tag.manifest.json ] && { echo "skip $id $tag"; continue; }
  /venv/dabai/bin/python scripts/quality/candidate_video_stage.py --id $id --tag $tag --manifest /workspace/dub/canonical/artifacts/baseline/$id/final/${id}_final.manifest.json \
    --out $O --work-out $O/work_${id}_$tag --mux final_mix --gate-merge-gap-s 1.2 --mouth-only-pasteback on --mask-smooth 3 --ls-landmark-smooth 2 \
    --codeformer-scope mouth --codeformer off --face-stack mediapipe "$@" > $O/${id}_$tag.log 2>&1 || echo "FAIL $id $tag"
  grep "^== " $O/${id}_$tag.log
done
