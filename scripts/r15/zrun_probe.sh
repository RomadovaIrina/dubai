#!/bin/bash
# r15 Z audit probe (CPU) on the q2d work dirs (or the newest q2 arm of the video when q2d is not there yet)
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export OMP_NUM_THREADS=8; mkdir -p /workspace/dub/r15w/audit
for id in ${*:-m18 old03 m03}; do
  W=$(ls -d /workspace/dub/artifacts/r4m/r9g_q2_${id}_q2d/work_* 2>/dev/null | head -1)
  [ -n "$W" ] || W=$(ls -dt /workspace/dub/artifacts/r4m/r9g_q2_${id}_*/work_* 2>/dev/null | head -1)
  [ -n "$W" ] || { echo "== probe $id: no work dir"; continue; }
  python scripts/r15/audit_probe.py --work $W --json /workspace/dub/r15w/audit/probe_$id.json > /workspace/dub/r15w/audit/probe_$id.txt 2>&1
  echo "== probe $id $(basename $(dirname $W)) rc=$?"
done
echo PROBE_DONE
