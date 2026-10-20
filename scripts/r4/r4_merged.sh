#!/bin/bash
# Round 4 MERGED (built on the audio server while the video server is down): ONE command, all videos in order.
#   m2      = r4_video_v1 stack (v4 generator + r3fix composite minus --seam-lf 4, as published by the video agent) on audio pb1 (user's choice)
#   v4repro = the full v4 stack (r3fix default) on this server's v4 audio reproduction (r4a/v4repro, bit-identical to audio-r4b's code) = gate PREV
# then r4_gate.py score for both. GPU stages serialise on the gpu.lock, composites overlap.
#   bash scripts/r4/r4_merged.sh [ids]      -> /workspace/dub/artifacts/r4m/{m2,v4repro}/<id>_<tag>.mp4 + .manifest.json + .score.json
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
B=/workspace/dub/artifacts/r4m; AU=/workspace/dub/artifacts/r4a; mkdir -p $B/m2 $B/v4repro
V1="--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8"
one() {   # id tag audio_tag [FIX30_ARGS]
  id=$1; tag=$2; at=$3; O=$B/$tag
  FIX30_ARGS="${4:-}" bash scripts/r4/r4_video.sh $id $AU/$at/$id $AU/$at/$id/${id}_r4b.manifest.json $O $tag > $O/${id}.run.log 2>&1 || { echo "FAIL $tag $id"; return; }
  mkdir -p /workspace/dub/artifacts/r3d/timing; for f in $O/timing/${id}_*; do [ -e /workspace/dub/artifacts/r3d/timing/$(basename $f) ] || cp $f /workspace/dub/artifacts/r3d/timing/; done   # r4_gate's fixed lm path
  python scripts/r4/r4_gate.py score --label $tag --dir $O --ids $id > $O/${id}.score.log 2>&1 || echo "WARN score $tag $id"
  echo "== $tag $id $(date +%T)"
}
for id in ${*:-04 03 05 01 02}; do
  one $id m2 pb1 "$V1" &
  one $id v4repro v4repro "" &
  wait
done
echo R4_MERGED_DONE
