cd /workspace/dub/dubai; export FC_CMS=1 LS_LAYER_RGB=1 LS_CMS_READ=1 QUICK=1
B=/workspace/dub/artifacts/r4m; SL=$B/r9g_m03/timing/m03_source_lm_F.npz; SRC=/workspace/dub/r10/runs/m03/work/in/m03.MOV
for v in "e34 3/4 6.25" "e45 4/5 5.0"; do set -- $v
  bash scripts/r12/ls_arm.sh m03 $1 "" LS_RATE=$2 LS_RATE_EVEN=1
  ( . scripts/env.sh >/dev/null
    for hz in $3 8.333; do python scripts/r12/mouth_pop.py --video $B/r9f_m03$1/m03_r9f_m03$1.mp4 --src $SRC --src-lm $SL --label "m03 $1 @${hz}Hz" --cad-hz $hz; done
    python scripts/r12/mouth_pop.py --video $B/r9f_m03cmsf/m03_r9f_m03cmsf.mp4 --src $SRC --src-lm $SL --label "m03 cmsf (control) @${3}Hz" --cad-hz $3 ) &
done; wait; echo EVEN2_DONE
