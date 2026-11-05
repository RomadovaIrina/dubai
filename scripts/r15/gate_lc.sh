#!/bin/bash
# r15 F2: after run_f2.sh, gate the m03 lip-contrast arms (ful / fubl / fucl) vs the production m03fu, same GPU rules; then the gate tables go
# into REPORT.md and DONE is touched.
set -u; R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd $R; . scripts/env.sh >/dev/null
L=/workspace/dub/r15f2/logs; D=/workspace/dub/r15f2
gpu_wait(){ while [ -e /workspace/dub/r15d/E2E_RUNNING ] || [ -e /workspace/dub/r15t/E2E_RUNNING ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -ge 2048 ]; do sleep 30; done; }
until grep -q F2_DONE $L/run_f2.out; do sleep 60; done
for sx in ful fubl fucl; do gpu_wait; GT=r9g_q2_m03fu bash scripts/r15/gate_f.sh q2_m03 $sx > $L/gate_q2_m03$sx.log 2>&1; tail -n 1 $L/gate_q2_m03$sx.log
  python scripts/r10/gate_cmp.py --base /workspace/dub/r15f/gate/q2_m03fu --cand /workspace/dub/r15f/gate/q2_m03$sx --md $D/gate.md | head -n 3; done
{ echo; echo "## Gates (filled automatically $(date +%F\ %T))"; echo; cat $D/gate.md; } >> $D/REPORT.md
cp $D/REPORT.md /workspace/dub/outbox/r15_f2/REPORT.md; touch $D/DONE; echo GATE_LC_DONE $(date +%T)
