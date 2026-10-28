#!/bin/bash
# Round 4 V: the v4 video stack (904c5d2 r3final v4) on a GIVEN audio dir, one video. No audio stage, no remix: the video is judged alone.
#   bash scripts/r4/r4_video.sh <id> <audio_dir> <audio_manifest> <out_dir> [tag]
# audio_dir: dubbed_16k.wav (generator input) + final_mix_aac.m4a (muxed).  Overrides (env): VIDEO_ARGS (generator), FIX30_ARGS (composite, full
# replacement of the r3fix default), MUX_AAC (an .m4a muxed onto the final instead of <audio_dir>/final_mix_aac.m4a, e.g. v4's own mix).
# Output: <out>/<id>_<tag>.mp4 + .manifest.json (stage wall times, peak VRAM). Timing / landmark caches per out dir (reused across arms).
set -uo pipefail
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . "${DUB_ROOT}/scripts/env.sh" >/dev/null; cd "${DUB_ROOT}"
id=$1; AD=$2; AM=$3; O=$4; TAG=${5:-r4v}; mkdir -p "$O"
in=$(ls /workspace/dub/inputs/test_videos_raw/${id}.* | head -1); W=$O/work_${id}_$TAG; rm -rf "$W"; mkdir -p "$W"; LOG=$O/${id}_${TAG}.log; : > "$LOG"
PC=/workspace/dub/artifacts/r3d/prep/$id   # phase 2: shared CPU-prep cache per source (25 fps master + its wav + MediaPipe passes; CleanPrep / make_master skip what exists)
if [ -n "${P2_PREP_CACHE:-}" ] && [ -d "$PC" ]; then mkdir -p "$W/video"; cp -n "$PC"/* "$W/video/" 2>/dev/null; echo "prep cache: $(ls $PC | tr '\n' ' ')" >> "$LOG"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500 > "$W/nvsmi.csv" 2>/dev/null & NV=$!
trap 'kill $NV 2>/dev/null' EXIT
t0=$(date +%s.%N)
GEN=${GEN_ARGS:-"--anchor-filter flow:4 --anchor-round off --affine-precision fp32 --mask-track head --master-timing-mode fps --guidance 2.0 --bridge-len 8 --bridge-steps 4 --max-ls-frames 400"}
[ -n "${LS_FPS:-}" ] && GEN="$GEN --ls-fps $LS_FPS"   # LatentSync on a master at LS_FPS (30 = the source frames; the composite then needs no retime)
flock /workspace/dub/gpu.lock python scripts/quality/candidate_video_stage.py --id $id --tag ${TAG}_25fps --manifest "$AM" --dubbed-dir "$AD" --out "$W" --work-out "$W/video" \
   --mux final_mix --gate-merge-gap-s 1.2 --mouth-only-pasteback on --mask-smooth 3 --ls-landmark-smooth 2 --codeformer-scope mouth --codeformer off \
   --face-stack mediapipe --dump-ls-raw on $GEN ${VIDEO_ARGS:-} >> "$LOG" 2>&1 || { echo "FAIL video $id (see $LOG)"; exit 4; }
VM=$W/${id}_${TAG}_25fps.manifest.json; t1=$(date +%s.%N)
if [ -n "${P2_PREP_CACHE:-}" ] && [ ! -d "$PC" ]; then mkdir -p "$PC"; cp "$W"/video/master_25fps.mp4 "$W"/video/*.wav "$W"/video/master_mediapipe*.npz "$W"/video/master_rigid.npz "$PC"/ 2>/dev/null; fi
if [ -n "${LS_SR_ARGS:-}" ]; then   # phase 2: Real-ESRGAN layer SR (r7_layer_sr.py) right after the generator, the composite then runs ONCE on the restored layer
  RAW=$W/${id}_${TAG}_25fps_ls_raw.mp4; SRL=$W/${id}_${TAG}_25fps_ls_srt1.mp4
  if [ -f "$RAW" ]; then   # the layer's MediaPipe pass (needed by the SR crop and by the composite) once, then the SR
    python -c "import sys, pathlib; sys.path.insert(0, 'scripts/pilot'); import clean_face_stack as cfs; cfs.run_mediapipe(pathlib.Path('$RAW'), pathlib.Path('${RAW%.mp4}.mediapipe.npz'))" >> "$LOG" 2>&1
    python scripts/r7/r7_layer_sr.py --ls-raw "$RAW" --lm "${RAW%.mp4}.mediapipe.npz" --out "$SRL" $LS_SR_ARGS >> "$LOG" 2>&1 && { cp "${RAW%.mp4}.mediapipe.npz" "${SRL%.mp4}.mediapipe.npz"; export FIX30_LS=$SRL; }
  fi
fi
mkdir -p "$O/timing"; TD=$O/timing; [ -n "${FIX30_TD_SHARED:-}" ] && { TD=/workspace/dub/artifacts/r3d/timing; mkdir -p $TD; }   # phase 2: shared source timing / landmark cache (keyed by id + master fps; source-only data)
FIX30_MASTER_FPS=${LS_FPS:-25} FIX30_TD=$TD FIX30_MODE=r3fix FIX30_THREADS=${FIX30_THREADS:-16} bash scripts/quality/r3_fix30.sh "$VM" final >> "$LOG" 2>&1
F=$W/${id}_${TAG}_25fps_final.mp4; FM=$W/${id}_${TAG}_25fps_final.manifest.json
[ -f "$F" ] || { echo "FAIL composite $id (see $LOG)"; exit 5; }
if [ -n "${MUX_AAC:-}" ]; then ffmpeg -loglevel error -y -i "$F" -i "$MUX_AAC" -map 0:v:0 -map 1:a:0 -c copy "$O/${id}_${TAG}.mp4"; else cp "$F" "$O/${id}_${TAG}.mp4"; fi
cp "$FM" "$O/${id}_${TAG}.manifest.json"; t2=$(date +%s.%N)
kill $NV 2>/dev/null; wait $NV 2>/dev/null
python - "$O/${id}_${TAG}.manifest.json" "$W/nvsmi.csv" "$t0" "$t1" "$t2" "$(git rev-parse --short HEAD)" "${MUX_AAC:-}" <<'PY'
import json, pathlib, sys
mp, nv = pathlib.Path(sys.argv[1]), sys.argv[2]; t = list(map(float, sys.argv[3:6])); m = json.loads(mp.read_text()); vmin = m["source"]["duration_s"] / 60
peak = max((int(l.split(",")[1]) for l in open(nv) if l.count(",") >= 2 and l.split(",")[1].strip().isdigit()), default=None)
st = {"latentsync_25fps": round(t[1] - t[0], 1), "composite_source_fps": round(t[2] - t[1], 1), "total": round(t[2] - t[0], 1)}
m["r4v"] = {"stage_wall_s": st, "s_per_video_min": {k: round(v / vmin, 1) for k, v in st.items()}, "peak_vram_mib": peak, "commit": sys.argv[6], "mux_aac": sys.argv[7] or None}
if sys.argv[7]: m["output"] = str(mp.with_suffix(".mp4"))
mp.write_text(json.dumps(m, indent=1)); print(f"== {mp.stem}: video {st['total']} s = {st['total'] / vmin:.0f} s/video-min; peak VRAM {peak} MiB")
PY
