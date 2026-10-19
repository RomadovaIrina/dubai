#!/bin/bash
# Round 3 D: FIX30 source-fps composite on a finished video-stage run that kept its raw layer (--dump-ls-raw on).
#   bash scripts/quality/r3_fix30.sh <video-stage manifest.json> [label suffix, default fix30]   (FIX30_ARGS env overrides the flags)
# -> <stem>_<suffix>.mp4 + .manifest.json (stage_seconds incl. fix30) + .fps_composite.json
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
M=$1; suf=${2:-fix30}; stem=${M%.manifest.json}; out=${stem}_$suf.mp4
FIX30_ARGS=${FIX30_ARGS:-"--register sim_ring --ring-region upper --edge-px 12"}
src=$(python -c "import json;print(json.load(open('$M'))['input'])"); id=$(basename ${src%.*}); T=/workspace/dub/artifacts/r3d/timing/${id}_master_timing.json
[ -f $T ] || python scripts/quality/master_timing.py --src $src --out $T
LS=${stem}_ls_raw.mp4; nls=$(python -c "import json;print(json.load(open('$M'))['lipsync']['by_action'].get('LATENT_SYNC',{}).get('frames',0))")
s=$(date +%s.%N)
if [ "$nls" -gt 0 ]; then
  python scripts/quality/fps_composite.py --manifest $M --ls-raw $LS --out $out --master-timing $T --threads ${FIX30_THREADS:-16} $FIX30_ARGS > ${stem}_$suf.log 2>&1 || echo "WARN fps_composite rc=$? $M"
else   # no LatentSync frames: the output IS the source video + the run's audio (copied)
  ffmpeg -loglevel error -y -i $src -i ${stem}.mp4 -map 0:v:0 -map 1:a:0 -c copy $out
fi
python - $M $out $s <<'PY'
import json, pathlib, sys, time
M, out, s = sys.argv[1], pathlib.Path(sys.argv[2]), float(sys.argv[3])
m = json.loads(pathlib.Path(M).read_text()); fj = out.with_suffix(".fps_composite.json"); f = json.loads(fj.read_text()) if fj.exists() else {"note": "no LS frames: source video copied"}
m["output_25fps"] = m["output"]; m["output"] = str(out); m["fix30"] = f; m["fix30"]["edge_px"] = (f.get("edge_px_measured") or {}).get("median", f.get("edge_px_target"))
st = dict(m["stage_seconds"]); st["fix30"] = round(time.time() - s, 2); st["total"] = round(st["total"] + st["fix30"], 2); m["stage_seconds"] = st
out.with_suffix(".manifest.json").write_text(json.dumps(m, indent=1)); print("==", out.name, "fix30", st["fix30"], "s", "fallback", f.get("fallback_fraction"), "edge", m["fix30"]["edge_px"])
PY
