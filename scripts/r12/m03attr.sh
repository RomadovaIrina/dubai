cd /workspace/dub/dubai; B=/workspace/dub/artifacts/r4m; Q2="--sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2"
gate() { DUB_INPUTS=/workspace/dub/r10/runs/$1/work/in bash scripts/r10/gate.sh $1 Y$2 $B/r9f_$1$2/$1_r9f_$1$2.mp4 > /workspace/dub/r14/logs/gate_$1$2.log 2>&1
  ( . scripts/env.sh >/dev/null; python scripts/r10/gate_cmp.py --base /workspace/dub/r10/gate/$1base --cand /workspace/dub/r10/gate/$1Y$2 --md /workspace/dub/r14/gate.md | head -n 3 ); }
gate m03 br16 &
( FCX="$Q2 --reg-smooth 4" bash scripts/r12/comp_k5.sh m03 rs4 FC_CMS=1 LS_CMS_READ=1 > /dev/null; gate m03 rs4 ) &
wait; echo M03ATTR_DONE
