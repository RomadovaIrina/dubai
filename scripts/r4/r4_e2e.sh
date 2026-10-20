#!/bin/bash
# Round 4 MERGED E2E (m3): ONE command, raw input -> final video, the same generic defaults for every video (no per-video flags).
#   audio : r4_audio.sh with the user's audio of record (pb1 = v1 + NE guard + source-speech bed duck -6 dB + pause boundary only)
#   video : r4_video_v1 stack (v4 generator + r3fix composite minus seam-LF) with LatentSync segments gated by the DUB speech (merge gap 0.4 s):
#           pb1's merged phrases left long dub silences inside 16 s segments (04 21.4-27.3 s: a quiet mouth painted over a talking face -> slide /
#           window joints); dub-gated segments end at those silences and the source frames show there (crossfade)
#   score : r4_gate.py score; compare against a v4repro set with r4_gate.py compare
#     bash scripts/r4/r4_e2e.sh [ids]     -> /workspace/dub/artifacts/r4m/m3/<id>_m3.mp4 (+ manifest, score), audio in /workspace/dub/artifacts/r4a/m3a
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
B=/workspace/dub/artifacts/r4m; AU=/workspace/dub/artifacts/r4a; TAG=${TAG:-m3}; AT=${TAG}a; mkdir -p $B/$TAG
AUDIO_FLAGS="--ne-guard on --pause-boundary-only on"; REMIX="--src-duck-db -6"
GATE="--gate-sources dub --gate-merge-gap-s 0.4"
V1="--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8"
for id in ${*:-04 03 05 01 02}; do
  t0=$(date +%s)
  PAR=1 AUDIO_EXTRA="$AUDIO_FLAGS" REMIX_ARGS="$REMIX" flock /workspace/dub/gpu.lock bash scripts/r4_audio.sh $AT $id > $B/$TAG/${id}.audio.log 2>&1
  [ -f $AU/$AT/$id/${id}_r4b.manifest.json ] || { echo "FAIL audio $id"; continue; }
  VIDEO_ARGS="$GATE" FIX30_ARGS="$V1" bash scripts/r4/r4_video.sh $id $AU/$AT/$id $AU/$AT/$id/${id}_r4b.manifest.json $B/$TAG $TAG > $B/$TAG/${id}.run.log 2>&1 || { echo "FAIL video $id"; continue; }
  mkdir -p /workspace/dub/artifacts/r3d/timing; for f in $B/$TAG/timing/${id}_*; do [ -e /workspace/dub/artifacts/r3d/timing/$(basename $f) ] || cp $f /workspace/dub/artifacts/r3d/timing/; done
  python scripts/r4/r4_gate.py score --label $TAG --dir $B/$TAG --ids $id > $B/$TAG/${id}.score.log 2>&1 || echo "WARN score $id"
  echo "== $TAG $id $(( $(date +%s) - t0 )) s $(date +%T)"
done
echo R4_E2E_DONE $TAG
