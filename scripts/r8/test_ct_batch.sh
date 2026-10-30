#!/bin/bash
# clause_takes on one video (the r8e2e0b retake output), sequential vs DUB_TTS_BATCH=1: the same layouts / acceptance, and the time
#   bash test_ct_batch.sh <id> <tag>     outputs /workspace/dub/r8a/try/ctb_<tag>_seq and _batch
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai
id=$1; tag=$2; A=/workspace/dub/artifacts/r4a
ARGS="--work $A/r8e2e0bsrc/rt/$id --src16 $A/r8e2e0bsrc/src16/${id}_16k.wav --face $A/sep/$id/source_mediapipe.npz"
for mode in seq batch; do O=/workspace/dub/r8a/try/ctb_${tag}_$mode; t0=$(date +%s)
  if [ $mode = batch ]; then export DUB_TTS_BATCH=1; else unset DUB_TTS_BATCH; fi
  python scripts/r8/clause_takes.py $ARGS --out $O > $O.log 2>&1; echo "$mode rc=$? $(( $(date +%s) - t0 )) s"; done
python3 - "$tag" <<'PY'
import json, sys
t = sys.argv[1]
a = json.load(open(f"/workspace/dub/r8a/try/ctb_{t}_seq/clause_takes.json")); b = json.load(open(f"/workspace/dub/r8a/try/ctb_{t}_batch/clause_takes.json"))
for x, y in zip(a, b):
    print(x["id"], "accepted", x["accepted"], y["accepted"], "| clauses same", x.get("clauses") == y.get("clauses"), "| cost", x.get("new_cost"), y.get("new_cost"), "| placed same", x.get("placed") == y.get("placed"))
PY
echo TEST_DONE
