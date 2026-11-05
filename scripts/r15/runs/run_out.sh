cd /workspace/dub/dubai-f; . scripts/env.sh >/dev/null
LCA=0.45 GT=r9g_q2_m03fu LT=both LT_EXTRA="--radius 5 --outlier 3" LT_TAG=r5o3 bash scripts/r15/comp_f2.sh q2_m03 fublab5o3 off > /dev/null 2>&1
LCA=0.45 GT=r9g_q2_m03fu LT=both LT_EXTRA="--radius 5 --outlier 2" LT_TAG=r5o2 bash scripts/r15/comp_f2.sh q2_m03 fublab5o2 off > /dev/null 2>&1
grep -h "lip_temporal" /workspace/dub/r15f2/logs/comp_q2_m03fublab5o*.log | cut -c1-300
B=/workspace/dub/artifacts/r4m; T=$B/r9g_q2_m03fu/timing; SRC=$(ls /workspace/dub/r14d/eval/q2_m03/e2e/work/in/q2_m03.* | head -1)
for t in fu fublab5 fublab5o3 fublab5o2; do python scripts/r12/mouth_pop.py --video $B/r9f_q2_m03$t/q2_m03_r9f_q2_m03$t.mp4 --src $SRC --src-lm $T/q2_m03_source_lm_F.npz --label $t | grep "^==" & done; wait
for t in fublab5o3 fublab5o2; do bash scripts/r15/lf.sh q2_m03$t /workspace/dub/r15f3/lipflick_w.jsonl | cut -c1-200; done
echo OUT_DONE
