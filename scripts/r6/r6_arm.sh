#!/bin/bash
# Round 6 F: composite-only ARM on m3's raw LatentSync layers (CPU; the generator output and the audio are reused), then the r4 gate score and the
# twitch score. Same pattern as r4_m4.sh.
#   EXTRA="<fps_composite flags>" bash scripts/r6/r6_arm.sh <arm> [ids]
#     -> /workspace/dub/artifacts/r6f/arms/<arm>/<id>_<arm>.mp4 + .manifest.json + score, twitch json in /workspace/dub/artifacts/r6f/twitch/<arm>/
# env: BASE (default m3), EXTRA (appended to the m3 V1 flag set), FIX30_THREADS (default 12)
set -u
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
ARM=$1; shift; IDS=${*:-05 04 03}; B=/workspace/dub/artifacts/r4m; BASE=${BASE:-m3}; O=/workspace/dub/artifacts/r6f/arms/$ARM; mkdir -p $O
V1="--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8"
echo "r6_arm $ARM commit $(git rev-parse --short HEAD) EXTRA='${EXTRA:-}'" >> $O/README.txt
for id in $IDS; do
  s=$(date +%s); VM=$B/$BASE/work_${id}_$BASE/${id}_${BASE}_25fps.manifest.json; [ -f $VM ] || { echo "no $BASE $id"; continue; }
  FIX30_TD=$B/$BASE/timing FIX30_MODE=r3fix FIX30_THREADS=${FIX30_THREADS:-12} FIX30_ARGS="$V1 ${EXTRA:-}" bash scripts/quality/r3_fix30.sh $VM $ARM > $O/${id}.log 2>&1
  F=${VM%.manifest.json}_$ARM.mp4; [ -f $F ] || { echo "FAIL $ARM $id"; tail -3 $O/${id}.log; continue; }
  mv $F $O/${id}_$ARM.mp4; mv ${F%.mp4}.fps_composite.json $O/${id}_$ARM.fps_composite.json 2>/dev/null
  python - ${F%.mp4}.manifest.json $O/${id}_$ARM.manifest.json $B/$BASE/${id}_$BASE.manifest.json "$ARM" <<'PY'
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text()); base = json.loads(pathlib.Path(sys.argv[3]).read_text()); out = pathlib.Path(sys.argv[2])
vmin = m["source"]["duration_s"] / 60; comp = m["stage_seconds"]["fix30"]; gen = base["r4v"]["stage_wall_s"]["latentsync_25fps"]
m["output"] = str(out.with_suffix("").with_suffix(".mp4")); m["r4v"] = {**base["r4v"], "arm": sys.argv[4], "stage_wall_s": {"latentsync_25fps": gen, "composite_source_fps": comp, "total": round(gen + comp, 1)}}
m["r4v"]["s_per_video_min"] = {k: round(v / vmin, 1) for k, v in m["r4v"]["stage_wall_s"].items()}
out.write_text(json.dumps(m, indent=1)); pathlib.Path(sys.argv[1]).unlink()
PY
  python scripts/r4/r4_gate.py score --label $ARM --dir $O --ids $id --asr-from $B/$BASE:$BASE > $O/${id}.score.log 2>&1 || echo "WARN score $id"
  MAN=$O/%s_$ARM.manifest.json bash scripts/r6/r6_twitch.sh $ARM $O _$ARM $id | grep '=='
  echo "== arm $ARM $id $(( $(date +%s) - s )) s $(date +%T)"
done
echo R6_ARM_DONE $ARM
