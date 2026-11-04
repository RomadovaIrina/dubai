cd /workspace/dub/dubai; B=/workspace/dub/artifacts/r4m; id=old03; O=/workspace/dub/outbox/r14_f; W="3-4,5-6,20.8-21,24,27-28"
Q2="--sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2"; F3="--reg-smooth 4 --mouth-region 0.45"
SRC=$(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1); SL=$B/r9g_$id/timing/${id}_source_lm_F.npz
P="--codeformer optimized --codeformer-w 1.0 --steps 10 --steps-silent 20 --codeformer-parse off --codeformer-batch 16"
meas() { t=$1; V=$B/r9f_$id$t/${id}_r9f_$id$t.mp4; ( . scripts/env.sh >/dev/null
  python scripts/r12/tone_flicker.py --video $V --src $SRC --src-lm $SL --label "$id $t" | tee -a /workspace/dub/r14/abl/$id.txt
  python scripts/r12/mouth_pop.py --video $V --src $SRC --src-lm $SL --label "$id $t" | tee -a /workspace/dub/r14/abl/${id}_pop.txt
  bash scripts/r12/clip_at.sh $id $B/r9f_${id}f3/${id}_r9f_${id}f3.mp4 "F3" $V "$t" "$W" $O/${id}_f3_vs_${t}_mouth05x.mp4 ) & }
# CPU: the source stubble closer to the lip line (composite only)
( FCX="$Q2 $F3 --lip-dilate 0.04" bash scripts/r12/comp_k5.sh $id f3t FC_CMS=1 LS_CMS_READ=1 > /dev/null; meas f3t; . scripts/env.sh >/dev/null
  flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $B/r9f_${id}f3t/${id}_r9f_${id}f3t.mp4 --audio /workspace/dub/artifacts/r4a/r9_${id}_htd/$id/dubbed_16k.wav --label "$id f3t" ) &
meas f3
# GPU probes (QUICK = SyncNet + NVML), composite = F3
export FC_CMS=1 LS_CMS_READ=1 QUICK=1 FC_EXTRA_ADD="$Q2 $F3"
bash scripts/r12/ls_arm.sh $id ov12c "" LS_STRIDE=12 LS_OVERLAP=continue; meas ov12c
bash scripts/r12/ls_arm.sh $id nd "$P --no-deepcache"; meas nd
bash scripts/r12/ls_arm.sh $id ov8f "$P --no-deepcache" LS_STRIDE=8 LS_OVERLAP=fuse; meas ov8f
wait; echo GEN1_DONE
