#!/bin/bash
# Round 7 MERGED E2E (r7 F): r6_e2e.sh + --paste-guard 0.5 (no paste when the current source frame has no face at the paste position) + the whole-frame gate.
#   face : r4_video_v1 stack + mouth-follows-dub rule (m3, r4_e2e.sh) + r6 F: sanitize alignment guard (fps_composite default) + --mouth-region 0.38 (m4)
#   audio: m5 = the audio agent's r6_audio_v1 (handoff/in/A/audio-r6-v1/<id>: dubbed_16k.wav + final_mix_aac.m4a + <id>_*.manifest.json)
#   score: r4_gate.py score; twitch r5_twitch.py
#     AUDIO=<root with <id>/ dirs> TAG=m5 bash scripts/r6/r6_e2e.sh [ids]   -> /workspace/dub/artifacts/r4m/<TAG>/<id>_<TAG>.mp4 (+ manifest, score)
#   phase 2 (speed): VIDEO_EXTRA="--steps 10 ..." (generator flags appended), FIX30_EXTRA="..." (composite flags appended, later wins)
set -u
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
B=/workspace/dub/artifacts/r4m; TAG=${TAG:-m6}; AUDIO=${AUDIO:-/workspace/dub/handoff/in/A/audio-r6-v1}; mkdir -p $B/$TAG
GATE="--gate-sources both --face-min-h 60 --ls-split-dub-silence 1.0 --cap-align on"
V1="--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8 --sanitize-guard on --mouth-region 0.38 --paste-guard ${PASTE_GUARD:-0.5}"
echo "r7_e2e $TAG commit $(git rev-parse --short HEAD) AUDIO=$AUDIO" >> $B/$TAG/README.txt
for id in ${*:-05 04 03 01 02}; do
  t0=$(date +%s); AD=$AUDIO/$id; AM=$(ls $AD/${id}_*.manifest.json 2>/dev/null | head -1)
  [ -f "$AD/dubbed_16k.wav" ] && [ -f "$AD/final_mix_aac.m4a" ] && [ -n "$AM" ] || { echo "FAIL audio $id (need $AD/dubbed_16k.wav, final_mix_aac.m4a, <id>_*.manifest.json)"; continue; }
  VIDEO_ARGS="$GATE ${VIDEO_EXTRA:-}" FIX30_ARGS="$V1 ${FIX30_EXTRA:-}" bash scripts/r4/r4_video.sh $id $AD $AM $B/$TAG $TAG > $B/$TAG/${id}.run.log 2>&1 || { echo "FAIL video $id"; continue; }
  mkdir -p /workspace/dub/artifacts/r3d/timing; for f in $B/$TAG/timing/${id}_*; do [ -e /workspace/dub/artifacts/r3d/timing/$(basename $f) ] || cp $f /workspace/dub/artifacts/r3d/timing/; done
  if [ -z "${SKIP_SCORE:-}" ]; then   # r7 F: SKIP_SCORE=1 when the run is only the generator stage of a later candidate (r7_m7_comp.sh scores that one)
  python scripts/r4/r4_gate.py score --label $TAG --dir $B/$TAG --ids $id > $B/$TAG/${id}.score.log 2>&1 || echo "WARN score $id"
  MAN=$B/$TAG/%s_$TAG.manifest.json bash scripts/r6/r6_twitch.sh $TAG $B/$TAG _$TAG $id | grep '=='
  bash scripts/r7/r7_frame_gate.sh $B/$TAG:$TAG $id | grep "^=="
  fi
  echo "== $TAG $id $(( $(date +%s) - t0 )) s $(date +%T)"
done
echo R7_E2E_DONE $TAG
