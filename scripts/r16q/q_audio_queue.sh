#!/bin/bash
# r16 Q: the accepted e2e audio (scripts/r16q/e2e_audio.sh) on each clip, one at a time, under gpu.lock (the audio stage holds 26-30 GB VRAM)
# usage: bash scripts/r16q/q_audio_queue.sh /workspace/dub/r16q/clips/q0*.mp4   -> /workspace/dub/r16q/e2e/<id>/
cd "$(dirname "$0")/../.."; set -a; . /workspace/dub/.env; set +a   # HF_TOKEN (gated pyannote)
for f in "$@"; do id=$(basename ${f%.*}); o=/workspace/dub/r16q/e2e/$id
  [ -f $o/${id}_DabAI_r9_spleeter.mp4 ] && continue
  echo "== $id $(date +%T)"; ID=$id flock /workspace/dub/gpu.lock bash scripts/r16q/e2e_audio.sh $f $o > /workspace/dub/r16q/e2e_$id.log 2>&1; tail -2 /workspace/dub/r16q/e2e_$id.log
done; echo Q_AUDIO_DONE
