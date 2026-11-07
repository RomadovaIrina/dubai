cd /workspace/dub/dubai
P="--codeformer optimized --codeformer-w 1.0 --steps 10 --steps-silent 20 --codeformer-parse off --codeformer-batch 16"
bash scripts/r12/ls_arm.sh m18 t7 "" LS_CFG_TRUNC=7
bash scripts/r12/ls_arm.sh m18 g1 "$P --guidance 1.0"
bash scripts/r12/ls_arm.sh m18 t5 "" LS_CFG_TRUNC=5
echo TRUNC2_DONE
