#!/bin/bash
# Round 6 MERGED E2E: ONE command, the m4 face stack on a GIVEN audio of record, every video the same flags (no per-video settings).
#   face : r4_video_v1 stack + mouth-follows-dub rule (m3, r4_e2e.sh) + r6 F: sanitize alignment guard (fps_composite default) + --mouth-region 0.38 (m4)
#   audio: m5 = the audio agent's r6_audio_v1 (handoff/in/A/audio-r6-v1/<id>: dubbed_16k.wav + final_mix_aac.m4a + <id>_*.manifest.json)
#   score: r4_gate.py score; twitch r5_twitch.py
#     AUDIO=<root with <id>/ dirs> TAG=m5 bash scripts/r6/r6_e2e.sh [ids]   -> /workspace/dub/artifacts/r4m/<TAG>/<id>_<TAG>.mp4 (+ manifest, score)
#   r7 block 9 (all default off): [--subs on|off] [--subs-lang src|dub] [--subs-style white|yellow|trendy|outline] [--watermark <png>] [--wm-corner tl|tr|bl|br] [--wm-opacity 0..1]
#     -> extra <id>_<TAG>_final.mp4 (burned subs/watermark, H.264 same res/fps, audio copied) via scripts/r7/r7_subs.py (CPU, 4 threads)
set -u
SUBS=off; SUBS_LANG=dub; SUBS_STYLE=yellow; WM=""; WM_CORNER=br; WM_OPACITY=0.6; IDS=()
while [ $# -gt 0 ]; do case "$1" in
  --subs) SUBS=$2; shift 2;; --subs-lang) SUBS_LANG=$2; shift 2;; --subs-style) SUBS_STYLE=$2; shift 2;;
  --watermark) WM=$2; shift 2;; --wm-corner) WM_CORNER=$2; shift 2;; --wm-opacity) WM_OPACITY=$2; shift 2;;
  *) IDS+=("$1"); shift;; esac; done
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
B=/workspace/dub/artifacts/r4m; TAG=${TAG:-m5}; AUDIO=${AUDIO:-/workspace/dub/handoff/in/A/audio-r6-v1}; mkdir -p $B/$TAG
GATE="--gate-sources both --face-min-h 60 --ls-split-dub-silence 1.0 --cap-align on"
V1="--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8 --sanitize-guard on --mouth-region 0.38"
echo "r6_e2e $TAG commit $(git rev-parse --short HEAD) AUDIO=$AUDIO" >> $B/$TAG/README.txt
for id in ${IDS[@]:-05 04 03 01 02}; do
  t0=$(date +%s); AD=$AUDIO/$id; AM=$(ls $AD/${id}_*.manifest.json 2>/dev/null | head -1)
  [ -f "$AD/dubbed_16k.wav" ] && [ -f "$AD/final_mix_aac.m4a" ] && [ -n "$AM" ] || { echo "FAIL audio $id (need $AD/dubbed_16k.wav, final_mix_aac.m4a, <id>_*.manifest.json)"; continue; }
  VIDEO_ARGS="$GATE" FIX30_ARGS="$V1" bash scripts/r4/r4_video.sh $id $AD $AM $B/$TAG $TAG > $B/$TAG/${id}.run.log 2>&1 || { echo "FAIL video $id"; continue; }
  mkdir -p /workspace/dub/artifacts/r3d/timing; for f in $B/$TAG/timing/${id}_*; do [ -e /workspace/dub/artifacts/r3d/timing/$(basename $f) ] || cp $f /workspace/dub/artifacts/r3d/timing/; done
  python scripts/r4/r4_gate.py score --label $TAG --dir $B/$TAG --ids $id > $B/$TAG/${id}.score.log 2>&1 || echo "WARN score $id"
  MAN=$B/$TAG/%s_$TAG.manifest.json bash scripts/r6/r6_twitch.sh $TAG $B/$TAG _$TAG $id | grep '=='
  if [ "$SUBS" = on ] || [ -n "$WM" ]; then
    [ "$SUBS" = on ] && SA=(--manifest $AM --lang $SUBS_LANG --style $SUBS_STYLE) || SA=(--no-subs)
    [ -n "$WM" ] && SA+=(--watermark $WM --wm-corner $WM_CORNER --wm-opacity $WM_OPACITY)
    python scripts/r7/r7_subs.py burn --video $B/$TAG/${id}_$TAG.mp4 "${SA[@]}" --out $B/$TAG/${id}_${TAG}_final.mp4 >> $B/$TAG/${id}.run.log 2>&1 || echo "WARN subs $id"
  fi
  echo "== $TAG $id $(( $(date +%s) - t0 )) s $(date +%T)"
done
echo R6_E2E_DONE $TAG
