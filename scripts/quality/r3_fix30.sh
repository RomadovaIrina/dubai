#!/bin/bash
# Round 3 D: FIX30 source-fps composite on a finished video-stage run that kept its raw layer (--dump-ls-raw on).
#   bash scripts/quality/r3_fix30.sh <video-stage manifest.json> [label suffix, default fix30]   (FIX30_ARGS env overrides the flags)
# -> <stem>_<suffix>.mp4 + .manifest.json (stage_seconds incl. fix30) + .fps_composite.json
set -u
DUB_ROOT=${DUB_ROOT:-/workspace/dub/dubai}; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
M=$1; suf=${2:-fix30}; stem=${M%.manifest.json}; out=${stem}_$suf.mp4
# FIX30_MODE=r2 (default): B round 2 T3u on a `-r 25` master.  FIX30_MODE=r3F: B round 3 (28491dd) on an fps master (arm r3F):
#   exact timing of THIS master (-vf fps=25), MediaPipe on every source frame, --source-lm + --colour-match upper --seam-lf 4 --sharpen-auto 1.0
MODE=${FIX30_MODE:-r2}
src=$(python -c "import json;print(json.load(open('$M'))['input'])"); id=$(basename ${src%.*}); TD=${FIX30_TD:-/workspace/dub/artifacts/r3d/timing}; mkdir -p $TD
WM=$(python -c "import json;print(json.load(open('$M'))['work_dir'])")/master_25fps.mp4
if [ $MODE = r3F ] || [ $MODE = r3cand ] || [ $MODE = r3fix ]; then
  if [ $MODE = r3fix ]; then   # r3cand + confidence-checked 25->30 retime (no torn / ghosted lips on interpolated frames, flash detector r3_flash.py)
    FIX30_ARGS=${FIX30_ARGS:-"--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --seam-lf 4 --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8"}
  elif [ $MODE = r3cand ]; then   # B's final R3 recipe (r3-face e9c3f58): + lip guard (8.4 on small faces), closed-loop sharpen to 0.95 + source skin texture
    FIX30_ARGS=${FIX30_ARGS:-"--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --seam-lf 4 --sharpen-auto 0.95 --skin-texture source --lip-guard on"}
  else
    FIX30_ARGS=${FIX30_ARGS:-"--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --seam-lf 4 --sharpen-auto 1.0"}
  fi
  T=$TD/${id}_master_timing_F.json; SL=$TD/${id}_source_lm_F.npz
  MF=${FIX30_MASTER_FPS:-25}; [ $MF = 25 ] || { T=$TD/${id}_master_timing_F$MF.json; SL=$TD/${id}_source_lm_F$MF.npz; FIX30_ARGS="$FIX30_ARGS --master-fps $MF"; }   # r4 V: LatentSync at the source rate
  [ -f $T ] || python scripts/quality/r3_master_timing.py --src $src --master $WM --vf fps=$MF --out $T
  [ -f $SL ] || python scripts/quality/r3_source_lm.py --src $src --timing $T --out $SL
  FIX30_ARGS="$FIX30_ARGS --source-lm $SL"
else
  FIX30_ARGS=${FIX30_ARGS:-"--register sim_ring --ring-region upper --edge-px 12"}
  T=$TD/${id}_master_timing.json
  [ -f $T ] || python scripts/quality/master_timing.py --src $src --out $T
fi
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
