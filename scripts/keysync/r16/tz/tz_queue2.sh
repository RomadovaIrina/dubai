#!/bin/bash
# r16 TZ, after tz_queue.sh (QUEUE_DONE): reproducibility of the dub (same raw input twice, audio only, default vs a deterministic-GPU harness)
# + a clean cold-start measurement (TensorRT cache + inductor cache set aside, one default job, then restored).
set -u; Q=/workspace/dub/r16tz; IN=$Q/in; L=$Q/queue.log; E="bash /workspace/dub/tzenv.sh"; S="$E /venv/dabai/bin/python scripts/keysync/r16/tz_suite.py"
log(){ echo "[$(date +%T)] $*" >> $L; }
until grep -q QUEUE_DONE $L 2>/dev/null; do sleep 30; done; log "queue2 start"
M03=/workspace/dub/inputs/mamakris/story-1774249919482-367803614.MOV
if [ ! -f $Q/repro.done ]; then
  mkdir -p $Q/repro/det; cat > $Q/repro/det/sitecustomize.py <<'PY'
# r16 TZ reproducibility harness (tests only): deterministic torch kernels in every python process of the job
try:
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
except Exception:
    pass
PY
  D="E.PYTHONPATH=$Q/repro/det,E.CUBLAS_WORKSPACE_CONFIG=:4096:8"
  $S --out $Q/repro --job r1=$M03,lipsync=off,keep=1 --job r2=$M03,lipsync=off,keep=1 --job d1=$M03,lipsync=off,keep=1,$D --job d2=$M03,lipsync=off,keep=1,$D \
     > $Q/repro/suite.log 2>&1
  for p in "r1 r2" "d1 d2" "r1 d1"; do set -- $p
    /venv/dabai/bin/python scripts/keysync/r16/tz/repro_diff.py /workspace/dub/artifacts/r4a/r9_tz_$1/tz_$1 /workspace/dub/artifacts/r4a/r9_tz_$2/tz_$2 \
      --out $Q/repro/diff_$1_$2.json > $Q/repro/diff_$1_$2.txt 2>&1; log "repro $1 vs $2: $(tail -1 $Q/repro/diff_$1_$2.txt)"; done
  touch $Q/repro.done
fi
if [ ! -f $Q/cold.done ]; then
  K=/workspace/dub/keysync; mkdir -p $Q/cold; mv $K/trt/cache $K/trt/cache_warm_bak; mv $K/inductor $K/inductor_warm_bak; log "cold: caches set aside"
  $S --out $Q/cold --job cold_m18=$IN/m18_20s.mp4 > $Q/cold/suite.log 2>&1
  grep -h "KS engines\|== KS" $Q/cold/cold_m18/e2e/work/ks/engines.log 2>/dev/null | tail -3 > $Q/cold/engines.txt
  python3 - $Q/cold > $Q/cold/summary.txt <<'PY'
import json, re, sys, pathlib
d = pathlib.Path(sys.argv[1]); s = json.load(open(d / "cold_m18" / "status.json")); log = (d / "cold_m18" / "e2e" / "logs" / "face_keysync.log").read_text(errors="replace")
t = {m.group(1): float(m.group(2)) for m in re.finditer(r"^KS_T (\S+) (\d+\.\d+)", log, re.M)}
k = list(t); seg = {k[i]: round(t[k[i + 1]] - t[k[i]], 1) for i in range(len(k) - 1)}
print(json.dumps({"status": s["status"], "cold_start": s.get("cold_start"), "wall_s": s.get("wall_s"), "error": s.get("error"), "ks_segments_s": seg}))
PY
  log "cold: $(cat $Q/cold/summary.txt | cut -c1-400)"
  if [ -d $K/trt/cache ] && ls $K/trt/cache/*/eng/kf/full.plan > /dev/null 2>&1; then rm -rf $K/trt/cache_warm_bak $K/inductor_warm_bak; log "cold: new caches kept"
  else rm -rf $K/trt/cache $K/inductor; mv $K/trt/cache_warm_bak $K/trt/cache; mv $K/inductor_warm_bak $K/inductor; log "cold: warm caches restored"; fi
  touch $Q/cold.done
fi
log "QUEUE2_DONE"
