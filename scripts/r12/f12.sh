cd /workspace/dub/dubai; B=/workspace/dub/artifacts/r4m; id=old03; Q2="--sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2"; SRC=$(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1); mkdir -p /workspace/dub/outbox/r14_f
run() { t=$1; fx=$2
  GT=r9g_${id}br16 FCX="$Q2 $fx" bash scripts/r12/comp_k5.sh $id $t FC_CMS=1 LS_CMS_READ=1 > /dev/null
  . scripts/env.sh >/dev/null; V=$B/r9f_$id$t/${id}_r9f_$id$t.mp4
  python scripts/r12/tone_flicker.py --video $V --src $SRC --src-lm $B/r9g_$id/timing/${id}_source_lm_F.npz --label "$id $t" | tee -a /workspace/dub/r14/abl/$id.txt
  flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $V --audio /workspace/dub/artifacts/r4a/r9_${id}_htd/$id/dubbed_16k.wav --label "$id $t"
  bash scripts/r12/clip_at.sh $id $B/r9f_$id/${id}_r9f_$id.mp4 "q2 (now)" $V "$t" "3-4,5-6,20.8-21,24,27-28" /workspace/dub/outbox/r14_f/${id}_q2_vs_${t}_mouth05x.mp4; }
run f1 "--reg-smooth 4" &
run f2 "--reg-smooth 4 --mouth-region 0.45" &
wait; echo F12_DONE
