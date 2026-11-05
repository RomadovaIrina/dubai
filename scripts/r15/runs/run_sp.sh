cd /workspace/dub/dubai-f; . scripts/env.sh >/dev/null
REPA="--spikes on" LCA=0.45 GT=r9g_q2_m03fu LT=both LT_EXTRA="--radius 5" LT_TAG=r5 bash scripts/r15/comp_f2.sh q2_m03 fublab5sp off | grep "layer_repair" | cut -c1-300
B=/workspace/dub/artifacts/r4m; T=$B/r9g_q2_m03fu/timing; SRC=$(ls /workspace/dub/r14d/eval/q2_m03/e2e/work/in/q2_m03.* | head -1)
python scripts/r12/mouth_pop.py --video $B/r9f_q2_m03fublab5sp/q2_m03_r9f_q2_m03fublab5sp.mp4 --src $SRC --src-lm $T/q2_m03_source_lm_F.npz --label sp | grep "^=="
bash scripts/r15/lf.sh q2_m03fublab5sp /workspace/dub/r15f3/lipflick_w.jsonl | cut -c1-200; echo SP_DONE
