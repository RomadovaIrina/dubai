cd /workspace/dub/dubai; export QUICK=1
B=/workspace/dub/artifacts/r4m; SL=$B/r9g_m03/timing/m03_source_lm_F.npz; SRC=/workspace/dub/r10/runs/m03/work/in/m03.MOV
for q in 0.5; do t=r23a${q#0.}
  bash scripts/r12/ls_arm.sh m03 $t "" LS_RATE=2/3 LS_RATE_ADAPT=$q
  grep -h "ls_rate\] adapt" $B/r9g_m03$t/*.log | head -2
  ( . scripts/env.sh >/dev/null; python scripts/r12/mouth_pop.py --video $B/r9f_m03$t/m03_r9f_m03$t.mp4 --src $SRC --src-lm $SL --label "m03 $t" )
done; echo ADAPT_DONE
