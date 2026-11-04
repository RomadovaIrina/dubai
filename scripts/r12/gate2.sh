cd /workspace/dub/dubai; B=/workspace/dub/artifacts/r4m
while pgrep -f "[p]robe_seg.py" > /dev/null; do sleep 15; done
gate() { DUB_INPUTS=/workspace/dub/r10/runs/$1/work/in flock /workspace/dub/gpu.lock bash scripts/r10/gate.sh $1 Y$2 $B/r9f_$1$2/$1_r9f_$1$2.mp4 > /workspace/dub/r14/logs/gate_$1$2.log 2>&1
  ( . scripts/env.sh >/dev/null; python scripts/r10/gate_cmp.py --base /workspace/dub/r10/gate/$1base --cand /workspace/dub/r10/gate/$1Y$2 --md /workspace/dub/r14/gate.md | head -n 3 ); }
gate old03 ov12c; gate old03 nd; echo GATE2_DONE
