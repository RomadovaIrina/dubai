cd /workspace/dub/dubai
until grep -q COMPILE_DONE /workspace/dub/r12/logs/compile.out; do sleep 20; done
export QUICK=1
B="--codeformer optimized --codeformer-w 1.0 --codeformer-parse off --codeformer-batch 16"
bash scripts/r12/ls_arm.sh m18 s8 "$B --steps 8 --steps-silent 20"
bash scripts/r12/ls_arm.sh m18 ss10 "$B --steps 10 --steps-silent 10"
bash scripts/r12/ls_arm.sh m18 r23 "" LS_RATE=2/3
echo STEPS_DONE
