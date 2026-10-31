#!/bin/bash
# the resident worker: <ids> through it in order (the first = warm-up tag r9wk0), each compared sample by sample with the cold round-2 run r9fast7
cd /workspace/dub/r9s
export DUB_UNIT_CONC=1 DUB_QWEN_BATCH=1 R9_MP_BG=on DUB_S3_BATCH=1 R9_ONEPROC=on R9_CHAIN_INPROC=on R9_SEP_OVERLAP=on R9_WORKER_SPOOL=/workspace/dub/r9s/worker
first=1
for id in "$@"; do t=r9wk; [ $first = 1 ] && t=r9wk0; first=0; bash gate_run.sh $t $id 0; done
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null 2>&1
python3 - "$@" <<PY
import sys, soundfile as sf, numpy as np
A = "/workspace/dub/artifacts/r4a"; ids = sys.argv[1:]
for k, id_ in enumerate(ids):
    t = "r9wk0" if k == 0 else "r9wk"; res = []
    for sub in (f"pre/runs/{id_}/work_{id_}", f"src/rt/{id_}", f"src/se/{id_}", f"/soft/{id_}"):
        a = f"{A}/r9fast7_{id_}{sub}/dubbed_24k.wav"; b = a.replace("r9fast7_", t + "_")
        try: x, _ = sf.read(a); y, _ = sf.read(b); res.append("=" if x.shape == y.shape and np.array_equal(x, y) else "DIFF")
        except Exception as e: res.append("ERR")
    print("WK_IDENT", t, id_, " ".join(res), "(pipeline / retake / soften / final dub)")
PY
echo WK_TEST_DONE
