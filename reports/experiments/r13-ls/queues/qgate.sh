cd /workspace/dub/dubai; B=/workspace/dub/artifacts/r4m
for a in "m03 cmsf" "m03 q" "old03 q"; do set -- $a; (
  bash scripts/r10/gate.sh $1 Y$2 $B/r9f_$1$2/$1_r9f_$1$2.mp4 > /workspace/dub/r12/logs/gate_$1$2.log 2>&1
  . scripts/env.sh >/dev/null; python scripts/r10/gate_cmp.py --base /workspace/dub/r10/gate/$1base --cand /workspace/dub/r10/gate/$1Y$2 --md /workspace/dub/r12/gate.md | head -n 3 ) &
done; wait; echo QGATE_DONE
