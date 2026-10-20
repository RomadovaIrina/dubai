cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=/workspace/dub/artifacts/r4sweep; AD=/workspace/dub/artifacts/r4a/cand2
for id in 04 05 03; do t=$(date +%s)
  python scripts/quality/candidate_video_stage.py --id $id --tag base_25fps --manifest $AD/$id/${id}_r4b.manifest.json --dubbed-dir $AD/$id --out $S --work-out $S/work_$id \
   --mux final_mix --gate-merge-gap-s 1.2 --mouth-only-pasteback on --mask-smooth 3 --ls-landmark-smooth 2 --codeformer-scope mouth --codeformer off \
   --face-stack mediapipe --dump-ls-raw on --anchor-filter flow:4 --anchor-round off --affine-precision fp32 --mask-track head --master-timing-mode fps \
   --guidance 2.0 --bridge-len 8 --bridge-steps 4 --max-ls-frames 400 > $S/base_$id.log 2>&1 && echo "== base $id $(( $(date +%s) - t )) s" || echo "FAIL base $id"
done
echo BASE_DONE
