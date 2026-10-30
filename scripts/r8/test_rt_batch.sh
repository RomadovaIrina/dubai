#!/bin/bash
# retake on one video (r8e2e0pre base), sequential vs DUB_TTS_BATCH=1: the same candidates / selection, and the time
#   bash test_rt_batch.sh <id> <tag>     outputs /workspace/dub/r8a/try/rtb_<tag>_seq and _batch (new names per tag, nothing removed)
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai PYTHONPATH=third_party/llama_cuda${PYTHONPATH:+:$PYTHONPATH}
id=$1; tag=$2; A=/workspace/dub/artifacts/r4a; W=$(readlink -f $A/r8e2e0pre/mix_src/$id)
ARGS="--work $W --face $A/sep/$id/source_mediapipe.npz --faith judge --src16 $A/r8e2e0bsrc/src16/${id}_16k.wav --cover-all on --short-s 0.3 --budget-scales 1.15,1.3,1.5,1.75 --cap-scale 1.2 --min-piece 0.2 --max-tts 4 --natural-cuts on --fillers on"
for mode in seq batch; do O=/workspace/dub/r8a/try/rtb_${tag}_$mode; t0=$(date +%s)
  if [ $mode = batch ]; then export DUB_TTS_BATCH=1; else unset DUB_TTS_BATCH; fi
  python scripts/r7/retake.py $ARGS --out $O > $O.log 2>&1; echo "$mode rc=$? $(( $(date +%s) - t0 )) s"; done
python3 - "$tag" <<'PY'
import json, sys
t = sys.argv[1]
a = json.load(open(f"/workspace/dub/r8a/try/rtb_{t}_seq/retake.json")); b = json.load(open(f"/workspace/dub/r8a/try/rtb_{t}_batch/retake.json"))
for x, y in zip(a["units"], b["units"]):
    print(x["id"], "accepted", x.get("accepted"), y.get("accepted"), "| same new text", x.get("new_en") == y.get("new_en"), "|", (x.get("new_en") or "")[:50], "||", (y.get("new_en") or "")[:50])
    ca = [(c.get("en"), c.get("cfg"), c.get("rejected")) for c in x["candidates"]]; cb = [(c.get("en"), c.get("cfg"), c.get("rejected")) for c in y["candidates"]]
    print("   candidates + verdicts identical:", ca == cb, len(ca), len(cb), "| costs", [c.get("cost") for c in x["candidates"] if c.get("cfg") is not None][:6], [c.get("cost") for c in y["candidates"] if c.get("cfg") is not None][:6])
PY
echo TEST_DONE
