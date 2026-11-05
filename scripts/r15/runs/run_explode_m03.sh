cd /workspace/dub/dubai-f; . scripts/env.sh >/dev/null
LCA=0.45 GT=r9g_q2_m03fu LT=both LT_EXTRA="--radius 5" LT_TAG=r5 bash scripts/r15/comp_f2.sh q2_m03 fublab5r off > /dev/null 2>&1
B=/workspace/dub/artifacts/r4m; W=$B/r9g_q2_m03fu/work_q2_m03_r9g_q2_m03fu; P=$W/q2_m03_r9g_q2_m03fu_25fps; T=$B/r9g_q2_m03fu/timing; SRC=$(ls /workspace/dub/r14d/eval/q2_m03/e2e/work/in/q2_m03.* | head -1); O=/workspace/dub/r15f3/explode; mkdir -p $O
python scripts/r15/explode.py --work $W --timing $T/q2_m03_master_timing_F.json --src $SRC --src-lm $T/q2_m03_source_lm_F.npz \
  --final recipe=$B/r9f_q2_m03fublab5/q2_m03_r9f_q2_m03fublab5.mp4 --final prod=$B/r9f_q2_m03fu/q2_m03_r9f_q2_m03fu.mp4 \
  --stage pre=${P}_ls_cfd_off.mp4 --stage sr=${P}_ls_cfd_off_srt1.mp4 --stage rep=${P}_ls_cfd_off_srt1_rep2slax.mp4 --stage lc2=${P}_ls_cfd_off_srt1_rep2slax_lc2.mp4 --stage lt=${P}_ls_cfd_off_srt1_rep2slax_lc2_ltbothr5.mp4 \
  --json $O/m03_recipe.json --sheet $O/m03_recipe_worst8.jpg > $O/m03_recipe.log 2>&1
python scripts/r15/explode.py --work $W --timing $T/q2_m03_master_timing_F.json --src $SRC --src-lm $T/q2_m03_source_lm_F.npz \
  --final prod=$B/r9f_q2_m03fu/q2_m03_r9f_q2_m03fu.mp4 --final recipe=$B/r9f_q2_m03fublab5/q2_m03_r9f_q2_m03fublab5.mp4 \
  --stage raw_cf=${P}_ls_raw.mp4 --stage cfd=${P}_ls_cfd_ia04.mp4 --stage sr=${P}_ls_cfd_ia04_srt1.mp4 --stage rep=${P}_ls_cfd_ia04_srt1_rep2s.mp4 \
  --json $O/m03_prod.json --sheet $O/m03_prod_worst8.jpg > $O/m03_prod.log 2>&1
echo EXPLODE_M03_DONE $(date +%T)
