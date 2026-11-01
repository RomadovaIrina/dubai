#!/bin/bash
until grep -q GWQ_DONE /workspace/dub/r10/after4.log; do sleep 30; done
bash /workspace/dub/r10/acct/gate_post.sh m03 _gw1 _gw1
cd /workspace/dub/dubai; python3 scripts/r9/gate_cmp.py --base /workspace/dub/r9/face/m03_b --cand /workspace/dub/r9/face/m03_gw1 2>&1 | tail -3
echo GWGATE_DONE
