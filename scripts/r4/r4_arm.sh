#!/bin/bash
# Round 4 V: one composite arm = the r4_video_v1 stack + extra fps_composite flags, re-composited (CPU) from a finished run's raw LatentSync layer.
#   bash scripts/r4/r4_arm.sh <id> <label> [extra fps_composite flags...]   -> $OUT_DIR/<id>_<label>.mp4 + .manifest.json, then gate score
# Env: OUT_DIR (default /workspace/dub/artifacts/r4v/arms), RUN_DIR / RUN_TAG (raw layer, default v4repro), NOSCORE=1 skips the score.
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
id=$1; L=$2; shift 2; B=/workspace/dub/artifacts/r4v; O=${OUT_DIR:-$B/arms}; RT=${RUN_TAG:-v4repro}; RD=${RUN_DIR:-$B/v4repro}; mkdir -p $O
VM=$RD/work_${id}_$RT/${id}_${RT}_25fps.manifest.json; [ -f $VM ] || { echo "no $RT run for $id"; exit 1; }
V1="--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8"
args="$V1 $*"
FIX30_TD=$B/v4repro/timing FIX30_MODE=r3fix FIX30_THREADS=${FIX30_THREADS:-12} FIX30_ARGS="$args" bash scripts/quality/r3_fix30.sh $VM $L > $O/${id}_$L.log 2>&1
F=${VM%.manifest.json}_$L.mp4; [ -f $F ] || { echo "FAIL $id $L"; exit 2; }
mv $F $O/${id}_$L.mp4; python - ${F%.mp4}.manifest.json $O/${id}_$L.manifest.json $RD/${id}_$RT.manifest.json "$L" "$args" <<'PY'
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text()); base = json.loads(pathlib.Path(sys.argv[3]).read_text()); out = pathlib.Path(sys.argv[2])
vmin = m["source"]["duration_s"] / 60; comp = m["stage_seconds"]["fix30"]; gen = base["r4v"]["stage_wall_s"]["latentsync_25fps"]
m["output"] = str(out.with_suffix("").with_suffix(".mp4")); m["r4v"] = {**base["r4v"], "arm": sys.argv[4], "fix30_args": sys.argv[5],
    "stage_wall_s": {"latentsync_25fps": gen, "composite_source_fps": comp, "total": round(gen + comp, 1)}}
m["r4v"]["s_per_video_min"] = {k: round(v / vmin, 1) for k, v in m["r4v"]["stage_wall_s"].items()}
out.write_text(json.dumps(m, indent=1)); pathlib.Path(sys.argv[1]).unlink()
PY
echo "== $id $L"
[ "${NOSCORE:-0}" = 1 ] || { python scripts/r4/r4_gate.py score --label $L --dir $O --ids $id --asr-from $RD:$RT > $O/${id}_$L.score.out 2>&1 && echo "== scored $id $L" || echo "FAIL score $id $L"; }
