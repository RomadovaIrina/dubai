cd /workspace/dub/dubai
until grep -q STEPS_DONE /workspace/dub/r12/logs/steps.out; do sleep 20; done
export QUICK=1
B="--codeformer optimized --codeformer-w 1.0 --codeformer-parse off --codeformer-batch 16"
bash scripts/r12/ls_arm.sh m18 s6 "$B --steps 6 --steps-silent 20"
bash scripts/r12/ls_arm.sh m18 s5 "$B --steps 5 --steps-silent 20"
bash scripts/r12/ls_arm.sh m18 i10 "$B --steps 10 --steps-silent 20 --deepcache-interval 10"
echo STEPS2_DONE
