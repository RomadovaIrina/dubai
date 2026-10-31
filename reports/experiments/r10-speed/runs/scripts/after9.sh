#!/bin/bash
until grep -q M01COMP2_DONE /workspace/dub/r10/after8.log; do sleep 30; done
cd /workspace/dub/dubai
/venv/dabai/bin/python scripts/r10/worker.py --queue /workspace/dub/r10/q4.txt --out /workspace/dub/r10/q4_i2_gwuc2 --inflight 2 --gen-worker --env E2E_METRICS=off LS_UNET_COMPILE=default 2>&1 | grep -E "QUEUE_DONE|done "
grep "gen_worker\]" /workspace/dub/r10/q4_i2_gwuc2/gen_spool/worker.log
echo GWUCQ2_DONE
