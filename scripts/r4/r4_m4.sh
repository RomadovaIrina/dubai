#!/bin/bash
# Round 4 MERGED m4 = m3 + fps_composite --mouth-region 0.38 (video agent's task-5 winner: 03 chin wave -30 %, 05 strong flashes 33 -> 23).
# Composite-only: m3's audio and raw LatentSync layer are reused (CPU), then scored (WER reused from m3: same audio).
#   bash scripts/r4/r4_m4.sh [ids]   -> /workspace/dub/artifacts/r4m/m4/<id>_m4.mp4 + manifest + score
set -u
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
B=/workspace/dub/artifacts/r4m; O=$B/m4; mkdir -p $O
V1="--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8"
for id in ${*:-05 04 03 01 02}; do
  VM=$B/m3/work_${id}_m3/${id}_m3_25fps.manifest.json; [ -f $VM ] || { echo "no m3 $id"; continue; }
  FIX30_TD=$B/m3/timing FIX30_MODE=r3fix FIX30_THREADS=${FIX30_THREADS:-10} FIX30_ARGS="$V1 --mouth-region 0.38" bash scripts/quality/r3_fix30.sh $VM m4 > $O/${id}.log 2>&1
  F=${VM%.manifest.json}_m4.mp4; [ -f $F ] || { echo "FAIL $id"; continue; }
  mv $F $O/${id}_m4.mp4; python - ${F%.mp4}.manifest.json $O/${id}_m4.manifest.json $B/m3/${id}_m3.manifest.json <<'PY'
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text()); base = json.loads(pathlib.Path(sys.argv[3]).read_text()); out = pathlib.Path(sys.argv[2])
vmin = m["source"]["duration_s"] / 60; comp = m["stage_seconds"]["fix30"]; gen = base["r4v"]["stage_wall_s"]["latentsync_25fps"]
m["output"] = str(out.with_suffix("").with_suffix(".mp4")); m["r4v"] = {**base["r4v"], "arm": "m4 = m3 + --mouth-region 0.38", "stage_wall_s": {"latentsync_25fps": gen, "composite_source_fps": comp, "total": round(gen + comp, 1)}}
m["r4v"]["s_per_video_min"] = {k: round(v / vmin, 1) for k, v in m["r4v"]["stage_wall_s"].items()}
out.write_text(json.dumps(m, indent=1)); pathlib.Path(sys.argv[1]).unlink()
PY
  python scripts/r4/r4_gate.py score --label m4 --dir $O --ids $id --asr-from $B/m3:m3 > $O/${id}.score.log 2>&1 || echo "WARN score $id"
  echo "== m4 $id $(date +%T)"
done
echo R4_M4_DONE
