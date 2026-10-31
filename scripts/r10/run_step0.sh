VX="--codeformer optimized --codeformer-w 1.0 --steps 20 --no-deepcache --codeformer-parse off --codeformer-batch 16"
for id in m18 m03 old03; do bash /workspace/dub/dubai/scripts/r10/run_gen_arm.sh $id s20 "$VX"; done
echo STEP0_GEN_DONE
