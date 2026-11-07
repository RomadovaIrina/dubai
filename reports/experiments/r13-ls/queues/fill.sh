cd /workspace/dub/dubai; export QUICK=1
B=/workspace/dub/artifacts/r4m; SL=$B/r9g_m03/timing/m03_source_lm_F.npz; SRC=/workspace/dub/r10/runs/m03/work/in/m03.MOV
for v in hf near; do
  bash scripts/r12/ls_arm.sh m03 r23$v "" LS_RATE=2/3 LS_RATE_FILL=$v
  ( . scripts/env.sh >/dev/null; python scripts/r12/mouth_pop.py --video $B/r9f_m03r23$v/m03_r9f_m03r23$v.mp4 --src $SRC --src-lm $SL --label "m03 r23$v" ) &
done; wait; echo FILL_DONE
