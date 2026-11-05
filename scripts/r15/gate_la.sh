#!/bin/bash
# r15 F2 round 2: gate the ADAPTIVE lip-contrast arms (m03 fubla / fucla vs fu, old03 bla / cla vs c0) after gate_lc.sh, same GPU rules.
set -u; R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd $R; . scripts/env.sh >/dev/null
L=/workspace/dub/r15f2/logs; D=/workspace/dub/r15f2
gpu_wait(){ while [ -e /workspace/dub/r15d/E2E_RUNNING ] || [ -e /workspace/dub/r15t/E2E_RUNNING ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -ge 2048 ]; do sleep 30; done; }
until [ -e $D/DONE ] && grep -q LA_DONE $L/run_la.out; do sleep 60; done
: > $D/gate_la.md
for x in q2_m03:r9g_q2_m03fu:fu:fubla q2_m03:r9g_q2_m03fu:fu:fucla q2_old03:r9g_q2_old03:c0:bla q2_old03:r9g_q2_old03:c0:cla; do IFS=: read id ST A sx <<< "$x"
  gpu_wait; GT=$ST bash scripts/r15/gate_f.sh $id $sx > $L/gate_$id$sx.log 2>&1; tail -n 1 $L/gate_$id$sx.log
  python scripts/r10/gate_cmp.py --base /workspace/dub/r15f/gate/$id$A --cand /workspace/dub/r15f/gate/$id$sx --md $D/gate_la.md | head -n 3; done
{ echo; echo "## Gates, adaptive lipcon arms ($(date +%F\ %T))"; echo; cat $D/gate_la.md; } >> $D/REPORT.md
cp $D/REPORT.md /workspace/dub/outbox/r15_f2/REPORT.md; touch $D/DONE_LA; echo GATE_LA_DONE $(date +%T)
