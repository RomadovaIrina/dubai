cd /workspace/dub/dubai-f
until grep -q LT3_DONE /workspace/dub/r15f3/run_lt3.out; do sleep 30; done
LCA=0.45 GT=r9g_q2_m18 LT=both LT_EXTRA="--radius 5" LT_TAG=r5 bash scripts/r15/comp_f2.sh q2_m18 blab5 off
bash scripts/r15/lf.sh q2_m18blab5 /workspace/dub/r15f3/lipflick_w.jsonl > /dev/null 2>&1; echo M18B_DONE $(date +%T)
