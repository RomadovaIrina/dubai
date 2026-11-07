cd /workspace/dub/dubai
until grep -q TRUNC3_DONE /workspace/dub/r12/logs/trunc3.out; do sleep 20; done
bash scripts/r12/ls_arm.sh m18 cd1 "" LS_CFG_DELTA=1
until grep -q R12_BASE_DONE /workspace/dub/r12/logs/base.out; do sleep 20; done
bash scripts/r12/ls_arm.sh m18 t0c ""
echo DELTA_DONE
