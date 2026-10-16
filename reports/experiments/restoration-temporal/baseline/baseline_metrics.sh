#!/bin/bash
# Full common metric set for one baseline run root (GPU idle otherwise): e2e table (+SyncNet mix / dub-only / original, WER, CER, coverage),
# face_temporal (switches, flicker, jitter, sharpness), pause_rhythm, audio_stages, md5 of the muxed audio stream.
set -u
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh
for id in "$@"; do
  R=/workspace/dub/artifacts/baseline/$id; M=$R/final/${id}_final.manifest.json
  echo "== metrics $id $(date -Iseconds)"
  python scripts/quality/e2e_report.py --run-root "$R" --ids "$id" --label baseline_cand5 --syncnet > "$R/metrics_e2e.log" 2>&1; echo "e2e rc=$?"
  if [ "$id" != "02" ]; then python scripts/quality/face_temporal.py --manifest "$M" --out "$R/face_temporal" --tag cand5 > "$R/metrics_face_temporal.log" 2>&1; echo "face_temporal rc=$?"; fi
  python scripts/quality/pause_rhythm.py --manifest "$M" --out "$R/pauses" --tag cand5 > "$R/metrics_pauses.log" 2>&1; echo "pauses rc=$?"
  python scripts/quality/audio_stages.py --manifest "$M" --out "$R/audio_stages" --tag cand5 > "$R/metrics_audio_stages.log" 2>&1; echo "audio_stages rc=$?"
  ffmpeg -loglevel error -i "$R/final/${id}_final.mp4" -map 0:a:0 -c copy -f md5 - > "$R/final/${id}_final.audio_stream.md5" 2>&1
  echo "== metrics $id done $(date -Iseconds)"
done
