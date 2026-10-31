#!/bin/bash
until grep -q GWUCQ2_DONE /workspace/dub/r10/after9.log; do sleep 30; done
cd /workspace/dub/dubai
/venv/dabai/bin/python scripts/r10/worker.py --queue /workspace/dub/r10/q3warm.txt --out /workspace/dub/r10/q3warm_dyn --inflight 2 --gen-worker --env E2E_METRICS=off LS_UNET_COMPILE=default LS_UNET_COMPILE_DYNAMIC=1 2>&1 | grep -E "QUEUE_DONE|done "
echo WARM_DONE
/venv/dabai/bin/python scripts/r10/worker.py --queue /workspace/dub/r10/q4u.txt --out /workspace/dub/r10/q4u_dyn --inflight 2 --gen-worker --env E2E_METRICS=off LS_UNET_COMPILE=default LS_UNET_COMPILE_DYNAMIC=1 2>&1 | grep -E "QUEUE_DONE|done "
grep "gen_worker\] job" /workspace/dub/r10/q4u_dyn/gen_spool/worker.log
echo Q4U_DONE
/venv/dabai/bin/python scripts/r10/worker.py --queue /workspace/dub/r10/q4u.txt --out /workspace/dub/r10/q4u_ref --inflight 2 --gen-worker --env E2E_METRICS=off 2>&1 | grep -E "QUEUE_DONE|done "
echo Q4UREF_DONE
