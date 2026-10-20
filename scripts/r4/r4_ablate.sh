#!/bin/bash
# Round 4 V task 2: one-layer-off ablation of the v4 composite (r3fix) on a finished v4-repro run. The generator output (raw LatentSync layer) is
# REUSED, only the source-fps composite re-runs (CPU), so every arm shares the same lips / audio and differs by that one layer alone.
#   bash scripts/r4/r4_ablate.sh <id> [arms...]   -> /workspace/dub/artifacts/r4v/abl/<id>_<arm>.mp4 + .manifest.json
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
id=$1; shift; B=/workspace/dub/artifacts/r4v; O=${OUT_DIR:-$B/abl}; mkdir -p $O; RT=${RUN_TAG:-v4repro}; RD=${RUN_DIR:-$B/v4repro}   # RUN_DIR / RUN_TAG: another run's raw layer (e.g. the merged m1)
VM=$RD/work_${id}_$RT/${id}_${RT}_25fps.manifest.json; [ -f $VM ] || { echo "no v4repro run for $id"; exit 1; }
FULL="--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --seam-lf 4 --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8"
declare -A DROP=( [no_retime]="--retime flow_safe=>--retime nearest" [no_sanitize]="--sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8=>" [no_sharpen]="--sharpen-auto 0.95=>|--sharpen-lips-max 0.8=>"
  [no_colour]="--colour-match upper=>" [no_skin]="--skin-texture source=>" [no_faceclip]="--face-clip on=>" [no_seamlf]="--seam-lf 4=>" [no_lipguard]="--lip-guard on=>"
  [v1]="--sharpen-auto 0.95=>|--sharpen-lips-max 0.8=>|--seam-lf 4=>" )   # r4 video v1 = v4 minus the two layers that never helped (ablation 01 / 03 / 04 / 05)
for arm in ${*:-no_retime no_sanitize no_sharpen no_colour no_skin no_faceclip no_seamlf no_lipguard}; do
  args=$FULL; IFS='|' read -ra subs <<< "${DROP[$arm]}"
  for s in "${subs[@]}"; do args=${args/"${s%%=>*}"/"${s#*=>}"}; done
  FIX30_TD=$B/v4repro/timing FIX30_MODE=r3fix FIX30_THREADS=${FIX30_THREADS:-12} FIX30_ARGS="$args" bash scripts/quality/r3_fix30.sh $VM $arm > $O/${id}_$arm.log 2>&1
  F=${VM%.manifest.json}_$arm.mp4; [ -f $F ] || { echo "FAIL $id $arm"; continue; }
  mv $F $O/${id}_$arm.mp4; python - ${F%.mp4}.manifest.json $O/${id}_$arm.manifest.json $RD/${id}_$RT.manifest.json "$arm" "$args" <<'PY'
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text()); base = json.loads(pathlib.Path(sys.argv[3]).read_text()); out = pathlib.Path(sys.argv[2])
vmin = m["source"]["duration_s"] / 60; comp = m["stage_seconds"]["fix30"]; gen = base["r4v"]["stage_wall_s"]["latentsync_25fps"]
m["output"] = str(out.with_suffix("").with_suffix(".mp4")); m["r4v"] = {**base["r4v"], "arm": sys.argv[4], "fix30_args": sys.argv[5],
    "stage_wall_s": {"latentsync_25fps": gen, "composite_source_fps": comp, "total": round(gen + comp, 1)}}
m["r4v"]["s_per_video_min"] = {k: round(v / vmin, 1) for k, v in m["r4v"]["stage_wall_s"].items()}
out.write_text(json.dumps(m, indent=1)); pathlib.Path(sys.argv[1]).unlink()
PY
  echo "== $id $arm"
done
