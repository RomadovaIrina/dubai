cd /workspace/dub/dubai
until grep -q TRUNC_M18_DONE /workspace/dub/r12/logs/trunc_m18.out; do sleep 20; done
bash scripts/r12/ls_arm.sh m18 t0b ""
echo T0B_DONE
