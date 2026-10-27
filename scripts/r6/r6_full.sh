#!/bin/bash
# r6 G: one generator arm on FULL videos = m4 (m3 E2E video stack + --mouth-region 0.38 composite) on m3's audio, only the generator changed;
# then r4_gate.py score (SyncNet dub speech, strong flashes, chin wave, mouth / dub mismatch, s/video-min, VRAM) + r5_twitch.py.
#   ARM=<name> GEN="<generator flags>" [CFG=<DUB_LS_CFG>] bash scripts/r6/r6_full.sh [ids]   -> /workspace/dub/artifacts/r6g/full/<ARM>/<id>_<ARM>.mp4
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
AU=/workspace/dub/artifacts/r4a/m3a; O=/workspace/dub/artifacts/r6g/full/$ARM; mkdir -p $O
GATE="--gate-sources both --face-min-h 60 --ls-split-dub-silence 1.0 --cap-align on"
V1="--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8 --mouth-region 0.38"
BASEGEN="--anchor-filter flow:4 --anchor-round off --affine-precision fp32 --mask-track head --master-timing-mode fps"
for id in ${*:-04 05}; do
  t0=$(date +%s)
  if [ ! -f $O/${id}_$ARM.mp4 ]; then
    DUB_LS_CFG="${CFG:-}" GEN_ARGS="$BASEGEN $GEN" VIDEO_ARGS="$GATE" FIX30_ARGS="${FIX30:-$V1}" bash scripts/r4/r4_video.sh $id $AU/$id $AU/$id/${id}_r4b.manifest.json $O $ARM > $O/${id}.run.log 2>&1 || { echo "FAIL video $id $ARM"; continue; }
  fi
  mkdir -p /workspace/dub/artifacts/r3d/timing; for f in $O/timing/${id}_*; do [ -e /workspace/dub/artifacts/r3d/timing/$(basename $f) ] || cp $f /workspace/dub/artifacts/r3d/timing/; done
  python scripts/r4/r4_gate.py score --label $ARM --dir $O --ids $id --asr-from /workspace/dub/artifacts/r4m/m3:m3 > $O/${id}.score.log 2>&1 || echo "WARN score $id"
  src=$(ls /workspace/dub/inputs/test_videos_raw/${id}.* | head -1)
  python scripts/r5/r5_twitch.py score --video $O/${id}_$ARM.mp4 --src $src --lm /workspace/dub/artifacts/r3d/timing/${id}_source_lm_F.npz --manifest $O/${id}_$ARM.manifest.json --out $O/${id}_$ARM.twitch.json > $O/${id}.twitch.log 2>&1 || echo "WARN twitch $id"
  echo "== $ARM $id $(( $(date +%s) - t0 )) s $(date +%T)"
done
echo R6_FULL_DONE $ARM
