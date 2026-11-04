cd /workspace/dub/dubai
for id in m03 old03; do
  flock /workspace/dub/gpu.lock env CL=ia04L FBL=fb7sL RL=rep2sL bash scripts/r12/comp_k5.sh $id cmsL FC_CMS=1 LS_LAYER_RGB=1
  ( . scripts/env.sh >/dev/null; SRC=$(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1); B=/workspace/dub/artifacts/r4m
    python scripts/r12/lab_drift.py --video $B/r9f_${id}cmsL/${id}_r9f_${id}cmsL.mp4 --src $SRC --src-lm $B/r9g_$id/timing/${id}_source_lm_F.npz --label "$id k5+CMS+layerRGB"
    flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $B/r9f_${id}cmsL/${id}_r9f_${id}cmsL.mp4 --audio /workspace/dub/artifacts/r4a/r9_${id}_htd/$id/dubbed_16k.wav --label "$id cmsL" ) &
done; wait; echo CMSL_DONE
