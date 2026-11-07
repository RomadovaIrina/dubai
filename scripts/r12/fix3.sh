cd /workspace/dub/dubai
until grep -q FIX2B_DONE /workspace/dub/r14/logs/fix2b.out; do sleep 20; done
B=/workspace/dub/artifacts/r4m; id=old03; Q2="--sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2"; SRC=$(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1)
run() { t=$1; shift; fx=$1; shift
  FCX="$Q2 $fx" bash scripts/r12/comp_k5.sh $id $t FC_CMS=1 LS_CMS_READ=1 "$@" > /dev/null
  . scripts/env.sh >/dev/null; V=$B/r9f_$id$t/${id}_r9f_$id$t.mp4
  python scripts/r12/tone_flicker.py --video $V --src $SRC --src-lm $B/r9g_$id/timing/${id}_source_lm_F.npz --label "$id $t" | tee -a /workspace/dub/r14/abl/$id.txt
  flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $V --audio /workspace/dub/artifacts/r4a/r9_${id}_htd/$id/dubbed_16k.wav --label "$id $t"; }
run rs4 "--reg-smooth 4" &
run rs8 "--reg-smooth 8" &
wait
run rs4dfk "--reg-smooth 4" DFK=dfk15 "DFK_ARGS=--sigma 1.5 --lf on"
echo FIX3_DONE
