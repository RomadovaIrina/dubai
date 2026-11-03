cd /workspace/dub/dubai; export FC_CMS=1 LS_LAYER_RGB=1 LS_CMS_READ=1 QUICK=1
for id in m03 old03; do
  bash scripts/r12/ls_arm.sh $id cmsf ""
  ( . scripts/env.sh >/dev/null; SRC=$(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1); B=/workspace/dub/artifacts/r4m
    python scripts/r12/lab_drift.py --video $B/r9f_${id}cmsf/${id}_r9f_${id}cmsf.mp4 --src $SRC --src-lm $B/r9g_$id/timing/${id}_source_lm_F.npz --label "$id k5+full CMS" ) &
done; wait; echo CMSF_DONE
