#!/bin/bash
# r7 F: composite-only A/B on a finished r7_e2e.sh run (its raw LatentSync layer is kept): the m6 composite flags + EXTRA (later flags win),
# then the seam / blur metric and the twitch detector on the result. No generator rerun: ~2.5 min per arm on 03.
#   EXTRA="--tex-tau 30" bash scripts/r7/r7_comp_arm.sh <arm> [id=03] [tag=m6]
#   -> /workspace/dub/artifacts/r7f/arms/<arm>/<id>_<arm>.mp4 (+ .fps_composite.json), seam json, twitch json; one '== arm' summary line
set -u
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
ARM=$1; id=${2:-03}; TAG=${3:-m6}; B=/workspace/dub/artifacts/r4m/$TAG; W=$B/work_${id}_$TAG; O=/workspace/dub/artifacts/r7f/arms/$ARM; mkdir -p $O
V1="--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8 --sanitize-guard on --mouth-region 0.38 --paste-guard 0.5"
src=$(ls /workspace/dub/inputs/test_videos_raw/$id.* | head -1)
s=$(date +%s)
FIX30_MASTER_FPS=25 FIX30_TD=$B/timing FIX30_MODE=r3fix FIX30_THREADS=${FIX30_THREADS:-12} FIX30_ARGS="$V1 ${EXTRA:-}" bash scripts/quality/r3_fix30.sh $W/${id}_${TAG}_25fps.manifest.json $ARM > $O/${id}.fix30.log 2>&1
[ -f $W/${id}_${TAG}_25fps_$ARM.mp4 ] || { echo "FAIL arm $ARM $id (see $O/${id}.fix30.log)"; exit 1; }
mv $W/${id}_${TAG}_25fps_$ARM.mp4 $O/${id}_$ARM.mp4; mv $W/${id}_${TAG}_25fps_$ARM.fps_composite.json $O/${id}_$ARM.fps_composite.json 2>/dev/null; mv $W/${id}_${TAG}_25fps_$ARM.manifest.json $O/${id}_$ARM.manifest.json 2>/dev/null; mv $W/${id}_${TAG}_25fps_$ARM.log $O/${id}_$ARM.log 2>/dev/null
t=$(( $(date +%s) - s ))
python scripts/r7/r7_seam.py --src $src --out-video $O/${id}_$ARM.mp4 --lm /workspace/dub/artifacts/r7f/lm/${id}_src_mp.npz --out $O/${id}_$ARM.seam.json --label $ARM --id $id --sheet $O/${id}_${ARM}_seam_sheet.jpg --top 6 | grep "^=="
OMP_NUM_THREADS=8 python scripts/r5/r5_twitch.py score --video $O/${id}_$ARM.mp4 --src $src --lm /workspace/dub/artifacts/r6f/lm/${id}_src_lm.npz --manifest $O/${id}_$ARM.manifest.json --out $O/${id}_$ARM.twitch.json > $O/${id}_$ARM.twitch.log 2>&1
echo "== arm $ARM $id composite $t s; twitch $(tail -1 $O/${id}_$ARM.twitch.log | python -c 'import json,sys; d=json.load(sys.stdin); print({k:d[k] for k in ("twitch_sum","n_events","top20_mean","p99")})' 2>/dev/null)"
