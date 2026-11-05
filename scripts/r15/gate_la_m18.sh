#!/bin/bash
# r15 F2: gate m18 b + adaptive lipcon (bla) vs production c0 after gate_la.sh, same GPU rules
set -u; R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd $R; . scripts/env.sh >/dev/null; L=/workspace/dub/r15f2/logs; D=/workspace/dub/r15f2
gpu_wait(){ while [ -e /workspace/dub/r15d/E2E_RUNNING ] || [ -e /workspace/dub/r15t/E2E_RUNNING ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -ge 2048 ]; do sleep 30; done; }
until [ -e $D/DONE_LA ] && [ -s /workspace/dub/artifacts/r4m/r9f_q2_m18bla/q2_m18_r9f_q2_m18bla.mp4 ]; do sleep 60; done
gpu_wait; GT=r9g_q2_m18 bash scripts/r15/gate_f.sh q2_m18 bla > $L/gate_q2_m18bla.log 2>&1; tail -n 1 $L/gate_q2_m18bla.log
python scripts/r10/gate_cmp.py --base /workspace/dub/r15f/gate/q2_m18c0 --cand /workspace/dub/r15f/gate/q2_m18bla --md $D/gate_la.md | head -n 3
{ echo; echo "## Gate m18 b + adaptive lipcon ($(date +%F\ %T))"; echo; tail -n 30 $D/gate_la.md; } >> $D/REPORT.md; cp $D/REPORT.md /workspace/dub/outbox/r15_f2/REPORT.md; touch $D/DONE_M18; echo GATE_M18_DONE
