cd /workspace/dub/dubai
until grep -q TRUNC2_DONE /workspace/dub/r12/logs/trunc2.out; do sleep 20; done
P="--codeformer optimized --codeformer-w 1.0 --steps 10 --steps-silent 20 --codeformer-parse off --codeformer-batch 16"
bash scripts/r12/ls_arm.sh m18 t7nd "$P --no-deepcache" LS_CFG_TRUNC=7
bash scripts/r12/ls_arm.sh m18 nd "$P --no-deepcache"
echo TRUNC3_DONE
