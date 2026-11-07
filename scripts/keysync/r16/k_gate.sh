#!/bin/bash
# r16 K1: gate SyncNet of a 25 fps video vs a dub wav: k_gate.sh <video> <dub16k.wav> <out.json>  -> prints global av_offset/conf + speech@0 mean
V=$(realpath $1); A=$(realpath $2); O=$(realpath -m $3); cd /workspace/dub/dubai; . scripts/env.sh >/dev/null 2>&1; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video $V --audio $A --out $O > $O.log 2>&1 || { echo "GATE_FAIL $V"; tail -3 $O.log; exit 1; }
python3 - $O <<"PY"
import json, sys
s = json.load(open(sys.argv[1])); sp = [p["conf_at0"] for p in s["per_second"] if p.get("conf_at0") is not None and (p.get("dub_act") or 0) >= 0.5]
print(json.dumps({"file": sys.argv[1], "global": s["global"], "speech_at0": round(sum(sp) / max(len(sp), 1), 3), "n_sec": len(sp)}))
PY
