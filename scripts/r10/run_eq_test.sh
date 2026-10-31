#!/bin/bash
# bash eq_test.sh <id> [extra key=value set for recomp]  -> md5 / RSS / time of fps_composite.py vs fps_composite_stream.py on the fx args
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ALLOW_NC=1
id=$1; SET=${2:-base}; B=/workspace/dub/artifacts/r4m; T=/workspace/dub/r10/eq; mkdir -p $T
A=$(python -c "import json;print(json.dumps(json.load(open('$B/r9f_$id/${id}_r9f_$id.fps_composite.json'))['args']))")
for v in ${EQ_V:-fps_composite_inmem fps_composite}; do
  python - "$A" "$SET" $T/${id}_${SET//[=,]/_}_$v.mp4 scripts/quality/$v.py <<'PY'
import json, resource, subprocess, sys, time
args = json.loads(sys.argv[1]); st = sys.argv[2]; out = sys.argv[3]; script = sys.argv[4]
if st != "base":
    for kv in st.split(","): k, v = kv.split("="); args[k] = v
args["out"] = out; args["threads"] = 16
cmd = [sys.executable, script]
for k, v in args.items():
    if v is None or v == "" or v is False: continue
    cmd += [f"--{k.replace('_', '-')}", str(v)]
t0 = time.time(); r = subprocess.run(cmd, capture_output=True, text=True)
print(f"{script.split('/')[-1]:28s} rc {r.returncode} {time.time() - t0:6.1f} s  peak RSS {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024} MB")
if r.returncode not in (0, 3): print(r.stdout[-1500:], r.stderr[-3000:])
PY
  echo "   md5 $(ffmpeg -loglevel error -i $T/${id}_${SET//[=,]/_}_$v.mp4 -map 0:v -f md5 - 2>&1)"
done
