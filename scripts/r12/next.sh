cd /workspace/dub/dubai; B=/workspace/dub/artifacts/r4m; Q2="--sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2"
gate() { DUB_INPUTS=/workspace/dub/r10/runs/$1/work/in bash scripts/r10/gate.sh $1 Y$2 $B/r9f_$1$2/$1_r9f_$1$2.mp4 > /workspace/dub/r14/logs/gate_$1$2.log 2>&1
  ( . scripts/env.sh >/dev/null; python scripts/r10/gate_cmp.py --base /workspace/dub/r10/gate/$1base --cand /workspace/dub/r10/gate/$1Y$2 --md /workspace/dub/r14/gate.md | head -n 3 ); }
gate old03 f1 & gate old03 f2 &
# m03: the stronger-bridge generator arm, then F1 / F2 composites
( export FC_CMS=1 LS_CMS_READ=1 QUICK=1 FC_EXTRA_ADD="$Q2"
  bash scripts/r12/ls_arm.sh m03 br16 "--codeformer optimized --codeformer-w 1.0 --steps 10 --steps-silent 20 --codeformer-parse off --codeformer-batch 16 --bridge-len 16 --bridge-steps 6" )
SRC=$(ls /workspace/dub/r10/runs/m03/work/in/m03.* | head -1)
for v in "f1|--reg-smooth 4" "f2|--reg-smooth 4 --mouth-region 0.45"; do t=${v%%|*}; x=${v#*|}; (
  GT=r9g_m03br16 FCX="$Q2 $x" bash scripts/r12/comp_k5.sh m03 $t FC_CMS=1 LS_CMS_READ=1 > /dev/null
  . scripts/env.sh >/dev/null; V=$B/r9f_m03$t/m03_r9f_m03$t.mp4
  python scripts/r12/tone_flicker.py --video $V --src $SRC --src-lm $B/r9g_m03/timing/m03_source_lm_F.npz --label "m03 $t" | tee -a /workspace/dub/r14/abl/m03.txt
  flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $V --audio /workspace/dub/artifacts/r4a/r9_m03_htd/m03/dubbed_16k.wav --label "m03 $t"
  gate m03 $t ) & done
( . scripts/env.sh >/dev/null; python scripts/r12/tone_flicker.py --video $B/r9f_m03/m03_r9f_m03.mp4 --src $SRC --src-lm $B/r9g_m03/timing/m03_source_lm_F.npz --label "m03 q2" | tee -a /workspace/dub/r14/abl/m03.txt )
wait; echo NEXT_DONE
