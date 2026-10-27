#!/bin/bash
# r6 G: composite-only arm = the raw LatentSync layer of a finished r6_full.sh arm re-composited (CPU) with extra fps_composite flags, scored.
#   SRC=F1_g25 ARM=F1d15 EXTRA="--deflash-mc 1.5" bash scripts/r6/r6_recomp.sh [ids]   -> /workspace/dub/artifacts/r6g/full/<ARM>/<id>_<ARM>.mp4
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
F=/workspace/dub/artifacts/r6g/full; O=$F/$ARM; mkdir -p $O
V1="${V1OVR:---register sim_ring --ring-region upper --edge-px 12 --colour-match upper --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8 --mouth-region 0.38}"
for id in ${*:-04 05}; do
  VM=$F/$SRC/work_${id}_$SRC/${id}_${SRC}_25fps.manifest.json; [ -f $VM ] || { echo "no $SRC $id"; continue; }
  FIX30_TD=$F/$SRC/timing FIX30_MODE=r3fix FIX30_THREADS=${FIX30_THREADS:-8} FIX30_ARGS="$V1 $EXTRA" bash scripts/quality/r3_fix30.sh $VM $ARM > $O/${id}.log 2>&1
  out=${VM%.manifest.json}_$ARM.mp4; [ -f $out ] || { echo "FAIL $ARM $id"; continue; }
  mv $out $O/${id}_$ARM.mp4; python - ${out%.mp4}.manifest.json $O/${id}_$ARM.manifest.json $F/$SRC/${id}_$SRC.manifest.json "$EXTRA" <<'PY'
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text()); base = json.loads(pathlib.Path(sys.argv[3]).read_text()); out = pathlib.Path(sys.argv[2])
m["output"] = str(out.with_suffix("").with_suffix(".mp4")); m["r4v"] = {**base["r4v"], "arm": f"recomposite + {sys.argv[4]}"}
out.write_text(json.dumps(m, indent=1)); pathlib.Path(sys.argv[1]).unlink()
PY
  grep -h "sanitize-layer" $O/${id}.log | tail -1
  python scripts/r4/r4_gate.py score --label $ARM --dir $O --ids $id --asr-from /workspace/dub/artifacts/r4m/m3:m3 > $O/${id}.score.log 2>&1 || echo "WARN score $id"
  src=$(ls /workspace/dub/inputs/test_videos_raw/${id}.* | head -1)
  python scripts/r5/r5_twitch.py score --video $O/${id}_$ARM.mp4 --src $src --lm /workspace/dub/artifacts/r3d/timing/${id}_source_lm_F.npz --manifest $O/${id}_$ARM.manifest.json --out $O/${id}_$ARM.twitch.json > $O/${id}.twitch.log 2>&1 || echo "WARN twitch $id"
  echo "== $ARM $id $(date +%T) $(tail -1 $O/${id}.score.log)"
done
echo R6_RECOMP_DONE $ARM
