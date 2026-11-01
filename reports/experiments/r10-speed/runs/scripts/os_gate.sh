#!/bin/bash
until grep -q RATE_DONE /workspace/dub/r10/acct/rate_gate.log && grep -q OSTAB_DONE /workspace/dub/r10/acct/ostab.log; do sleep 20; done
cd /workspace/dub/r10/acct; bash gate_post.sh m03 _bos _b & bash gate_post.sh m03 _hos _h; wait
cd /workspace/dub/dubai; python3 scripts/r9/gate_cmp.py --base /workspace/dub/r9/face/m03_bos --cand /workspace/dub/r9/face/m03_hos 2>&1 | tail -3
echo OSGATE_DONE
