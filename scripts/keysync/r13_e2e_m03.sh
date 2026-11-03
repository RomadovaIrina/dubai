#!/bin/bash
# r13: the m03 E2E, held back while the urgent old03 KeySync final had the GPU
cd /workspace/dub/dubai; while pgrep -f "scripts/r9/e2e\.sh" > /dev/null; do sleep 20; done
M=/workspace/dub/inputs/mamakris; t0=$(date +%s)
ID=m03 E2E_METRICS=off bash scripts/r9/e2e.sh $M/story-1774249919482-367803614.MOV /workspace/dub/r13k/runs/m03 > /workspace/dub/r13k/e2e_m03.log 2>&1; echo "E2E m03 rc=$? $(( $(date +%s) - t0 ))s"
echo R13_E2E_DONE
