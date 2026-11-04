cd /workspace/dub/dubai
for id in m03 old03; do (
  GT=r9g_${id}cmsf FCX="--sharpen-auto 1.15 --sharpen-lips-max 1.5" bash scripts/r12/comp_k5.sh $id q FC_CMS=1 LS_LAYER_RGB=1 LS_CMS_READ=1
  . scripts/env.sh >/dev/null; SRC=$(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1); B=/workspace/dub/artifacts/r4m
  python scripts/r12/lab_drift.py --video $B/r9f_${id}q/${id}_r9f_${id}q.mp4 --src $SRC --src-lm $B/r9g_$id/timing/${id}_source_lm_F.npz --label "$id k5+quality"
  flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $B/r9f_${id}q/${id}_r9f_${id}q.mp4 --audio /workspace/dub/artifacts/r4a/r9_${id}_htd/$id/dubbed_16k.wav --label "$id q"
  bash scripts/r12/clip_ls.sh $id q "k5 + colour + detail" ) &
done; wait; echo QSTACK_DONE
