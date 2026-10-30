#!/bin/bash
# DabAI round-3 E2E runner (Agent D, v5): ONE command, raw video in -> final mp4 out, the same generic defaults for every video.
#   audio : Silero VAD -> faster-whisper -> pyannote -> Qwen GGUF (GPU) -> Chatterbox (speakable text, lexicon) -> cut at energy minima past
#           the CTC word end / merge voiced cuts / equal-power fades -> placement (r2 policy + SENTENCE_FLOW_GAP) -> Spleeter bed (GPU)
#           -> bed remix (duck -2 dB / 0.4 / 0.8 s, residue -6 dB) + static loudness (-16 LUFS, oversampled TP limiter)
#   video : B's on-time master (-vf fps=25) -> MediaPipe face stack -> A's R3 generator (fp32 AlignRestore, flow:4 anchors, head mask,
#           guidance 2.0, bridge 8x4, 400-frame calls) -> source-fps composite (B's final + flow_safe retime, generator-outlier sanitizer
#           (texture mode, relative threshold), lip sharpen cap, face clip) -> mux
#   QA    : r3_qa_gates.py on the output (frames / fps / res, lip flash, background ring, audio hard cuts, TTS joins, loudness) -> manifest
#
#   bash scripts/r3_e2e.sh <id|input.mp4> [<out_dir>]          (TAG=e2e_v5 by default; SENTENCE_FLOW_GAP default: see below)
# Output: <out>/<id>_<TAG>.mp4 + .manifest.json (stage wall times, peak VRAM, qa_gates). Nothing is reused from earlier runs (own work dir,
# own timing / landmark caches), so the stage times are a real cold E2E.
set -uo pipefail
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; export DUB_ROOT; . "${DUB_ROOT}/scripts/env.sh" >/dev/null; cd "${DUB_ROOT}"
arg=$1; O=${2:-/workspace/dub/artifacts/r3d/e2e_v5}; mkdir -p "$O"
if [ -f "$arg" ]; then in=$arg; id=$(basename "${arg%.*}"); else id=$arg; in=$(ls ${DUB_INPUTS:-/workspace/dub/inputs/test_videos_raw}/${id}.* | head -1); fi
TAG=${TAG:-e2e_v5}; W=$O/work_$id; rm -rf "$W"; mkdir -p "$W"; LOG=$O/${id}_${TAG}.log; : > "$LOG"
FLOW_GAP=${SENTENCE_FLOW_GAP:-1.0}   # generic default (03 / 04 A/B: joins inside sentences 15 / 10 -> 7 / 7, source-speaks-dub-silent +1.8 / +2.2 s; 3.0 would add +6-7.5 s)
export PYTHONPATH=third_party/llama_cuda${PYTHONPATH:+:$PYTHONPATH} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DUB_CUT_SNAP=1 DUB_EQPOWER_FADE=1 DUB_MERGE_VOICED_CUTS=1
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500 > "$W/nvsmi.csv" 2>/dev/null & NV=$!
trap 'kill $NV 2>/dev/null' EXIT
t0=$(date +%s.%N)
# ---- 1. audio (dub voice + placement)
python scripts/pilot/run_clean_pipeline_05.py --input "$in" --output "$W/${id}_audio.mp4" --target-lang en --alignment burst --video-backend optimized \
   --codeformer off --face-stack mediapipe --stop-after audio --audio-policy r2 --sentence-flow-gap $FLOW_GAP --qwen-gpu-layers -1 --tts-text-norm on \
   --qa-whisper-compute int8_float16 --work-dir "$W/audio" ${AUDIO_ARGS:-} >> "$LOG" 2>&1 || { echo "FAIL audio $id (see $LOG)"; exit 2; }
AW=$W/audio/work_$(basename "${in%.*}"); cp "$W/${id}_audio.manifest.json" "$AW/${id}_audio.manifest.json"
t1=$(date +%s.%N)
# ---- 2. bed remix + static loudness (same dub voice)
mkdir -p "$W/mix_src"; ln -sfn "$AW" "$W/mix_src/$id"
python scripts/quality/r3_remix.py --src "$W/mix_src" --out "$W/mix" $id >> "$LOG" 2>&1 || { echo "FAIL remix $id"; exit 3; }
MD=$W/mix/$id; MM=$MD/${id}_r4b.manifest.json
t2=$(date +%s.%N)
# ---- 3. LatentSync (A's R3 generator on B's on-time master), raw layer kept
python scripts/quality/candidate_video_stage.py --id $id --tag ${TAG}_25fps --manifest "$MM" --dubbed-dir "$MD" --out "$W" --work-out "$W/video" \
   --mux final_mix --gate-merge-gap-s 1.2 --mouth-only-pasteback on --mask-smooth 3 --ls-landmark-smooth 2 --codeformer-scope mouth --codeformer off \
   --face-stack mediapipe --anchor-filter flow:4 --anchor-round off --affine-precision fp32 --mask-track head --dump-ls-raw on --master-timing-mode fps \
   --guidance 2.0 --bridge-len 8 --bridge-steps 4 --max-ls-frames 400 ${VIDEO_ARGS:-} >> "$LOG" 2>&1 || { echo "FAIL video $id (see $LOG)"; exit 4; }
VM=$W/${id}_${TAG}_25fps.manifest.json
t3=$(date +%s.%N)
# ---- 4. source-fps composite (generic r3fix defaults)
FIX30_TD=$W/timing FIX30_MODE=r3fix FIX30_THREADS=${FIX30_THREADS:-12} bash scripts/quality/r3_fix30.sh "$VM" final >> "$LOG" 2>&1
F=$W/${id}_${TAG}_25fps_final.mp4; FM=$W/${id}_${TAG}_25fps_final.manifest.json
[ -f "$F" ] || { echo "FAIL composite $id (see $LOG)"; exit 5; }
cp "$F" "$O/${id}_${TAG}.mp4"; cp "$FM" "$O/${id}_${TAG}.manifest.json"
t4=$(date +%s.%N)
kill $NV 2>/dev/null; wait $NV 2>/dev/null
# ---- 5. QA gates + stage times / peak VRAM into the manifest
LM=$W/timing/${id}_source_lm_F.npz; [ -f "$LM" ] || python scripts/quality/r3_source_lm.py --src "$in" --timing "$(ls $W/timing/${id}_master_timing_F.json 2>/dev/null || echo none)" --out "$LM" >> "$LOG" 2>&1
python scripts/quality/r3_qa_gates.py --final "$O/${id}_${TAG}.mp4" --manifest "$O/${id}_${TAG}.manifest.json" --src "$in" --lm "$LM" --audio-manifest "$AW/${id}_audio.manifest.json" 2>&1 | tail -2 | tee -a "$LOG"
python - "$O/${id}_${TAG}.manifest.json" "$W/nvsmi.csv" "$t0" "$t1" "$t2" "$t3" "$t4" "$AW/${id}_audio.manifest.json" <<'PY'
import json, pathlib, sys
mp, nv = pathlib.Path(sys.argv[1]), sys.argv[2]; t = list(map(float, sys.argv[3:8])); am = json.loads(pathlib.Path(sys.argv[8]).read_text())
m = json.loads(mp.read_text()); vmin = m["source"]["duration_s"] / 60
peak = max((int(l.split(",")[1]) for l in open(nv) if l.count(",") >= 2 and l.split(",")[1].strip().isdigit()), default=None)
st = {"audio": round(t[1] - t[0], 1), "bed_remix_loudness": round(t[2] - t[1], 1), "latentsync_25fps": round(t[3] - t[2], 1), "composite_source_fps": round(t[4] - t[3], 1)}
st["total"] = round(t[4] - t[0], 1)
m["e2e"] = {"stage_wall_s": st, "s_per_video_min": {k: round(v / vmin, 1) for k, v in st.items()}, "peak_vram_mib": peak, "audio_stage_seconds": am.get("stage_seconds"),
            "defaults": "r3_e2e.sh v5 (no per-video flags)"}
mp.write_text(json.dumps(m, indent=1))
print(f"== {mp.stem}: total {st['total']} s = {st['total'] / vmin:.0f} s/video-min (audio {st['audio'] / vmin:.0f}, LatentSync {st['latentsync_25fps'] / vmin:.0f}, "
      f"composite {st['composite_source_fps'] / vmin:.0f}); peak VRAM {peak} MiB")
PY
