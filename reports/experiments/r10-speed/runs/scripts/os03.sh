#!/bin/bash
cd /workspace/dub/r10/acct; bash ostab.sh 03:_b 03:_r23
bash gate_post.sh 03 _bos _b & bash gate_post.sh 03 _r23os _r23; wait
cd /workspace/dub/dubai; python3 scripts/r9/gate_cmp.py --base /workspace/dub/r9/face/03_bos --cand /workspace/dub/r9/face/03_r23os 2>&1 | tail -3
echo OS03_DONE
