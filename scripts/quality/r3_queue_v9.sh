#!/bin/bash
# r3 D queue v9: the GPU audit (after the runner guard fix) -> E2E timing. Waits for running scorecards; scorers pause between items while it runs.
cd /workspace/dub/dubai
while pgrep -f "r3_scorecard[.]py score" >/dev/null; do sleep 15; done
bash scripts/quality/r3_gpu_audit.sh gpu 03 04; echo "AUDIT2_GPU_DONE $(date +%T)"
bash scripts/quality/r3_gpu_audit.sh cpuqwen 04; echo "AUDIT2_CPUQWEN_DONE $(date +%T)"
bash scripts/quality/r3_gpu_audit.sh resaudio 04 03; echo "AUDIT2_RESAUDIO_DONE $(date +%T)"
bash scripts/quality/r3_gpu_audit.sh resident 04 03; echo "AUDIT2_RESIDENT_DONE $(date +%T)"
TAG=r3e2e bash scripts/r3_e2e.sh 04 /workspace/dub/artifacts/r3d/e2e; echo E2E04_DONE
