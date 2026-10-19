#!/bin/bash
# Round 3 candidate E2E runner (Agent D): ONE command per video, every stage, the clean defaults + each track's best settings.
#   Silero VAD -> faster-whisper -> pyannote -> Qwen GGUF -> Chatterbox -> fit / place (C: --audio-policy r2, Spleeter bed, declick, duck)
#   -> LatentSync on the 25 fps master (A: fp32 AlignRestore, flow:4 anchors, sub-pixel, head-locked mask; raw layer kept)
#   -> FIX30 source-fps composite (B: fps_composite.py, true master timing, sim + upper ring registration, 12 px edge) -> mux (final mix copied)
#
#   bash scripts/r3_e2e.sh <id|input.mp4> [<out_dir>]
# Overrides (env):  AUDIO_ARGS   extra run_clean_pipeline_05.py flags (audio stage)          AUDIO_DIR  reuse a delivered audio dir (skip stage 1;
#                   VIDEO_ARGS   candidate_video_stage.py flags (default = A r2best)                    dubbed_16k.wav + final_mix_aac.m4a + manifest)
#                   FIX30_ARGS   fps_composite.py flags (default = B r2 T3u)                  FIX30=off  stop at the 25 fps output (old behaviour)
#                   BASE_MANIFEST  run manifest whose audio stages are reused with AUDIO_DIR (default: canonical cand5 of <id>)
# Output: <out_dir>/<id>_r3.mp4 (source fps / resolution), <id>_r3.manifest.json (stage_seconds of every stage), <id>_r3_25fps.mp4 (generator output).
set -uo pipefail
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export DUB_ROOT; . "${DUB_ROOT}/scripts/env.sh" >/dev/null; cd "${DUB_ROOT}"
arg=$1; O=${2:-/workspace/dub/artifacts/r3d/e2e}; mkdir -p "$O"
if [ -f "$arg" ]; then in=$arg; id=$(basename "${arg%.*}"); else id=$arg; in=$(ls /workspace/dub/inputs/test_videos_raw/${id}.* | head -1); fi
TAG=${TAG:-r3}; W=$O/work; mkdir -p "$W"; LOG=$O/${id}_${TAG}.log; : > "$LOG"
VIDEO_ARGS=${VIDEO_ARGS:-"--anchor-filter flow:4 --anchor-round off --affine-precision fp32 --mask-track head"}
FIX30_ARGS=${FIX30_ARGS:-"--register sim_ring --ring-region upper --edge-px 12"}
t0=$(date +%s.%N)
# ---- stage 1: audio (or a delivered audio dir)
if [ -n "${AUDIO_DIR:-}" ]; then
  AM=${BASE_MANIFEST:-/workspace/dub/canonical/artifacts/baseline/$id/final/${id}_final.manifest.json}; DUBDIR="--dubbed-dir $AUDIO_DIR"; echo "audio: delivered $AUDIO_DIR (stages of $AM)" | tee -a "$LOG"
else
  AM=$O/${id}_${TAG}_audio.manifest.json; DUBDIR=""
  # Qwen on the GPU (CUDA llama-cpp-python in third_party/llama_cuda, gpu_audit.md §5: audio stage 561 -> 122 s/video-min); QWEN_GPU=0 = CPU
  QG=""; if [ "${QWEN_GPU:-1}" = 1 ] && [ -f third_party/llama_cuda/llama_cpp/__init__.py ]; then export PYTHONPATH=third_party/llama_cuda${PYTHONPATH:+:$PYTHONPATH}; QG="--qwen-gpu-layers -1"; fi
  python scripts/pilot/run_clean_pipeline_05.py --input "$in" --output "$O/${id}_${TAG}_audio.mp4" --target-lang en --alignment burst --video-backend optimized \
     --codeformer off --face-stack mediapipe --work-dir "$W" --stop-after audio --audio-policy r2 $QG \
     ${AUDIO_ARGS:-} >> "$LOG" 2>&1 || { echo "FAIL audio $id (see $LOG)"; exit 2; }
  [ -f "$AM" ] || AM=$(ls -t $O/${id}_${TAG}_audio*.manifest.json 2>/dev/null | head -1)
fi
t1=$(date +%s.%N)
# ---- stage 2: LatentSync on the 25 fps master + raw layer
flock /workspace/dub/gpu.lock python scripts/quality/candidate_video_stage.py --id $id --tag ${TAG}_25fps --manifest "$AM" $DUBDIR --out "$O" --work-out "$W/video_$id" \
   --mux final_mix --gate-merge-gap-s 1.2 --mouth-only-pasteback on --mask-smooth 3 --ls-landmark-smooth 2 --codeformer-scope mouth --codeformer off --face-stack mediapipe \
   --dump-ls-raw $([ "${FIX30:-on}" = off ] && echo off || echo on) $VIDEO_ARGS >> "$LOG" 2>&1 || { echo "FAIL video $id (see $LOG)"; exit 3; }
VM=$O/${id}_${TAG}_25fps.manifest.json; t2=$(date +%s.%N)
# ---- stage 3: FIX30 source-fps composite
if [ "${FIX30:-on}" != off ]; then
  python scripts/quality/master_timing.py --src "$in" --out "$W/${id}_master_timing.json" --master "$W/video_$id/master_25fps.mp4" >> "$LOG" 2>&1
  python scripts/quality/fps_composite.py --manifest "$VM" --ls-raw "$O/${id}_${TAG}_25fps_ls_raw.mp4" --out "$O/${id}_${TAG}.mp4" --master-timing "$W/${id}_master_timing.json" \
     --threads ${FIX30_THREADS:-16} $FIX30_ARGS >> "$LOG" 2>&1; rc=$?
  [ $rc -eq 0 ] || echo "WARN fps_composite rc=$rc (3 = > 2 % fallback frames)" | tee -a "$LOG"
else cp "$O/${id}_${TAG}_25fps.mp4" "$O/${id}_${TAG}.mp4"; fi
t3=$(date +%s.%N)
python - "$AM" "$VM" "$O/${id}_${TAG}.mp4" "$t0" "$t1" "$t2" "$t3" "${AUDIO_DIR:-}" <<'PY'
import json, pathlib, sys
am, vm, out, t0, t1, t2, t3, adir = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]), *map(float, sys.argv[4:8]), sys.argv[8]
a, v = json.loads(pathlib.Path(am).read_text()), json.loads(pathlib.Path(vm).read_text())
st = {}
if not adir: st.update({f"audio.{k}": x for k, x in (a.get("stage_seconds") or {}).items() if k != "total"})
st.update({f"video.{k}": x for k, x in (v.get("stage_seconds") or {}).items() if k != "total"})
fj = out.with_suffix(".fps_composite.json"); fix = json.loads(fj.read_text()) if fj.exists() else None
if fix: st.update({f"fix30.{k}": x for k, x in (fix.get("seconds") or {}).items() if k != "total"})
st.update({"wall.audio": round(t1 - t0, 2), "wall.video_25fps": round(t2 - t1, 2), "wall.fix30": round(t3 - t2, 2), "total": round(t3 - t0, 2),
           "latentsync": (v.get("stage_seconds") or {}).get("latentsync")})
m = {**v, "output": str(out), "output_25fps": v["output"], "audio_manifest": am, "audio_dir": adir or None, "fix30": fix, "stage_seconds": st}
if fix: m["fix30"]["edge_px"] = (fix.get("edge_px_measured") or {}).get("median", fix.get("edge_px_target"))
out.with_suffix(".manifest.json").write_text(json.dumps(m, indent=1))
dur = v["source"]["duration_s"] / 60
print(f"== {out.name}: wall audio {t1-t0:.0f} s, video {t2-t1:.0f} s, fix30 {t3-t2:.0f} s, total {t3-t0:.0f} s = {(t3-t0)/dur:.0f} s/video-min")
PY
