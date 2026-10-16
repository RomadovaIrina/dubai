#!/bin/bash
# Track B Phase 4: the main line (fps_composite.py) on clean0 runs of every video + the full check set.
#   TAG=MAIN ARGS="--boundary on --skin-texture source --lip-texture on --ls-sharpen 1.5 --ls-sharpen-lips 1.5" bash scripts/quality/run_phase4.sh 03 01 04 05 02
# needs: clean0 run (candidate_video_stage.py, CLEAN0.md recipe) at $C/<id>_clean0.manifest.json and the clean decomposition S0 at
# $D/<id>/S0_ls_raw.mp4 (decompose_video_stage.py --frames all; for a video without LatentSync frames the clean0 output is passed instead)
set -u
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . "${DUB_ROOT}/scripts/env.sh"; cd "${DUB_ROOT}"
R=/workspace/dub/artifacts/restoration-temporal; C=$R/clean0; D=$R/cdecomp; O=$R/phase4; mkdir -p $O
TAG=${TAG:-MAIN}; ARGS=${ARGS:-"--boundary on --skin-texture source --lip-texture on --ls-sharpen 1.5 --ls-sharpen-lips 1.5"}
for id in "$@"; do
  M=$C/${id}_clean0.manifest.json; S0=$D/$id/S0_ls_raw.mp4; [ -f $S0 ] || S0=$C/${id}_clean0.mp4
  SRC=$(python -c "import json;print(json.load(open('$M'))['input'])"); BASE=/workspace/dub/artifacts/baseline/$id/final/${id}_final.mp4
  out=$O/${id}_${TAG}.mp4; s=$(date +%s)
  python scripts/quality/fps_composite.py --manifest $M --ls-raw $S0 --out $out $ARGS > $O/${id}_${TAG}.log 2>&1; rc=$?
  echo "== $id $TAG rc=$rc $(( $(date +%s) - s ))s $(tail -1 $O/${id}_${TAG}.log | cut -c1-240)"
  python - "$SRC" "$out" "$BASE" > $O/${id}_${TAG}.checks.json <<'PY'
import json, subprocess, sys
src, out, base = sys.argv[1:4]
def pr(p):
    o = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
                                   "-show_entries", "format=duration", "-of", "json", p], capture_output=True, text=True).stdout)
    s = o["streams"][0]; n, d = s["avg_frame_rate"].split("/")
    return {"w": s["width"], "h": s["height"], "fps": round(float(n) / float(d), 3), "frames": int(s["nb_read_frames"]), "duration": round(float(o["format"]["duration"]), 3)}
def md5(p): return subprocess.run(["ffmpeg", "-loglevel", "error", "-i", p, "-map", "0:a:0", "-c", "copy", "-f", "md5", "-"], capture_output=True, text=True).stdout.strip()
a, b = pr(src), pr(out)
r = {"source": a, "output": b, "resolution_equal": (a["w"], a["h"]) == (b["w"], b["h"]), "fps_equal": abs(a["fps"] - b["fps"]) < 0.05, "frames_equal": a["frames"] == b["frames"],
     "duration_delta_s": round(b["duration"] - a["duration"], 3), "audio_md5_identical_to_baseline": md5(out) == md5(base)}
r["pass"] = r["resolution_equal"] and r["fps_equal"] and r["frames_equal"] and abs(r["duration_delta_s"]) <= 0.15 and r["audio_md5_identical_to_baseline"]
print(json.dumps(r, indent=1))
PY
  python -c "import json;r=json.load(open('$O/${id}_${TAG}.checks.json'));print('   checks', 'PASS' if r['pass'] else 'FAIL', {k:r[k] for k in ('resolution_equal','fps_equal','frames_equal','duration_delta_s','audio_md5_identical_to_baseline')}, r['output']['fps'], 'fps')"
  python scripts/quality/mouth_jitter_hr.py --video $out --reference $SRC --manifest $M --out $O/${id}_${TAG}.mouth_jitter_hr.json --tag ${id}_${TAG} > $O/${id}_${TAG}.mjhr.log 2>&1 &
done
wait
for id in "$@"; do
  M=$C/${id}_clean0.manifest.json
  if python -c "import json,sys;m=json.load(open('$M'));sys.exit(0 if m['lipsync']['by_action'].get('LATENT_SYNC',{}).get('frames',0) else 1)"; then
    DUB=$(python -c "import json;print(json.load(open('$M'))['dubbed_track']['aac'])")
    flock /workspace/dub/gpu.lock python scripts/quality/syncnet_fps.py $O/${id}_${TAG}.mp4 --dub $DUB --json $O/${id}_${TAG}.syncnet.json 2>/dev/null | tail -1 | sed "s/^/   $id syncnet /"
  else echo "   $id syncnet N/A (no LatentSync frames)"; fi
done
echo PHASE4_DONE
