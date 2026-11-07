cd /workspace/dub/dubai; B=/workspace/dub/artifacts/r4m; id=old03; O=/workspace/dub/outbox/r14_f; W="3-4,5-6,20.8-21,24,27-28"
Q2="--sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2"; F3="--reg-smooth 4 --mouth-region 0.45"
SRC=$(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1); SL=$B/r9g_$id/timing/${id}_source_lm_F.npz
P="--codeformer optimized --codeformer-w 1.0 --steps 10 --steps-silent 20 --codeformer-parse off --codeformer-batch 16"
meas() { t=$1; V=$B/r9f_$id$t/${id}_r9f_$id$t.mp4; ( . scripts/env.sh >/dev/null
  python scripts/r12/tone_flicker.py --video $V --src $SRC --src-lm $SL --label "$id $t" | tee -a /workspace/dub/r14/abl/$id.txt
  python scripts/r12/mouth_pop.py --video $V --src $SRC --src-lm $SL --label "$id $t" | tee -a /workspace/dub/r14/abl/${id}_pop.txt
  bash scripts/r12/clip_at.sh $id $B/r9f_${id}f3/${id}_r9f_${id}f3.mp4 "F3" $V "$t" "$W" $O/${id}_f3_vs_${t}_mouth05x.mp4 ) & }
# CPU: the source stubble closer to the lip line (composite only)
export FC_CMS=1 LS_CMS_READ=1 QUICK=1 FC_EXTRA_ADD="$Q2 $F3"
while pgrep -f "[g]ate2.sh" > /dev/null; do sleep 15; done
bash scripts/r12/ls_arm.sh $id rc "$P --guidance 3.0 --ref-mode shift32"; meas rc; wait; echo GEN2_DONE
