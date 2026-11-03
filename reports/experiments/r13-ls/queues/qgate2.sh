cd /workspace/dub/dubai; B=/workspace/dub/artifacts/r4m
for id in m03 old03; do (
  bash scripts/r10/gate.sh $id Ycmsr $B/r9f_${id}cmsr/${id}_r9f_${id}cmsr.mp4 > /workspace/dub/r12/logs/gate_${id}cmsr.log 2>&1
  . scripts/env.sh >/dev/null; python scripts/r10/gate_cmp.py --base /workspace/dub/r10/gate/${id}base --cand /workspace/dub/r10/gate/${id}Ycmsr --md /workspace/dub/r12/gate.md | head -n 3 ) &
done; wait; echo QGATE2_DONE
