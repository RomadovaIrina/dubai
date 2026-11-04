cd /workspace/dub/dubai; B=/workspace/dub/artifacts/r4m; Q2="--sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2"; O=/workspace/dub/outbox/r14_f
gate() { DUB_INPUTS=/workspace/dub/r10/runs/$1/work/in bash scripts/r10/gate.sh $1 Y$2 $B/r9f_$1$2/$1_r9f_$1$2.mp4 > /workspace/dub/r14/logs/gate_$1$2.log 2>&1
  ( . scripts/env.sh >/dev/null; python scripts/r10/gate_cmp.py --base /workspace/dub/r10/gate/$1base --cand /workspace/dub/r10/gate/$1Y$2 --md /workspace/dub/r14/gate.md | head -n 3 ); }
declare -A W=([old03]="3-4,5-6,20.8-21,24,27-28" [m03]="8.3-8.8,11.3-11.7,24.2-24.8")
for id in old03 m03; do (
  FCX="$Q2 --reg-smooth 4 --mouth-region 0.45" bash scripts/r12/comp_k5.sh $id f3 FC_CMS=1 LS_CMS_READ=1 > /dev/null
  . scripts/env.sh >/dev/null; SRC=$(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1); V=$B/r9f_${id}f3/${id}_r9f_${id}f3.mp4
  python scripts/r12/tone_flicker.py --video $V --src $SRC --src-lm $B/r9g_$id/timing/${id}_source_lm_F.npz --label "$id f3" | tee -a /workspace/dub/r14/abl/$id.txt
  flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $V --audio /workspace/dub/artifacts/r4a/r9_${id}_htd/$id/dubbed_16k.wav --label "$id f3"
  bash scripts/r12/clip_at.sh $id $B/r9f_$id/${id}_r9f_$id.mp4 "q2 (now)" $V "f3" "${W[$id]}" $O/${id}_q2_vs_f3_mouth05x.mp4
  bash scripts/r12/clip_ls.sh $id f3 "q2 + f3" $O 4 | tail -1
  gate $id f3 ) &
done; wait; echo F3_DONE
