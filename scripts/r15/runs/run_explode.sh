cd /workspace/dub/dubai-f; . scripts/env.sh >/dev/null
until grep -q REGEN_DONE /workspace/dub/r15f3/run_regen.out; do sleep 20; done
B=/workspace/dub/artifacts/r4m; W=$B/r9g_q2_old03/work_q2_old03_r9g_q2_old03; P=$W/q2_old03_r9g_q2_old03_25fps; T=$B/r9g_q2_old03/timing; SRC=$(ls /workspace/dub/r14d/eval/q2_old03/e2e/work/in/q2_old03.* | head -1); O=/workspace/dub/r15f3/explode; mkdir -p $O
python scripts/r15/explode.py --work $W --timing $T/q2_old03_master_timing_F.json --src $SRC --src-lm $T/q2_old03_source_lm_F.npz \
  --final recipe=$B/r9f_q2_old03blab5/q2_old03_r9f_q2_old03blab5.mp4 --final prod=$B/r9f_q2_old03c0/q2_old03_r9f_q2_old03c0.mp4 \
  --stage pre=${P}_ls_cfd_off.mp4 --stage sr=${P}_ls_cfd_off_srt1.mp4 --stage rep=${P}_ls_cfd_off_srt1_rep2slax.mp4 --stage lc2=${P}_ls_cfd_off_srt1_rep2slax_lc2.mp4 --stage lt=${P}_ls_cfd_off_srt1_rep2slax_lc2_ltbothr5.mp4 \
  --json $O/old03_recipe.json --sheet $O/old03_recipe_worst8.jpg > $O/old03_recipe.log 2>&1
python scripts/r15/explode.py --work $W --timing $T/q2_old03_master_timing_F.json --src $SRC --src-lm $T/q2_old03_source_lm_F.npz \
  --final prod=$B/r9f_q2_old03c0/q2_old03_r9f_q2_old03c0.mp4 --final recipe=$B/r9f_q2_old03blab5/q2_old03_r9f_q2_old03blab5.mp4 \
  --stage raw_cf=${P}_ls_raw.mp4 --stage cfd=${P}_ls_cfd_ia04.mp4 --stage sr=${P}_ls_cfd_ia04_srt1.mp4 --stage rep=${P}_ls_cfd_ia04_srt1_rep2s.mp4 \
  --json $O/old03_prod.json > $O/old03_prod.log 2>&1
echo EXPLODE_DONE $(date +%T)
