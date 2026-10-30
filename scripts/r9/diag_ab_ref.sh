#!/bin/bash
# r9 diag A/B on one video: the m19b face with the LatentSync reference from another time (--ref-mode shift16 / fixed) instead of the frame
# itself (same), on the SAME dub. Generator + m19b composite (production only), then SyncNet per second on the dub and the source-copy check.
#   bash ab_ref.sh <id> <mode> ...     -> /workspace/dub/artifacts/r4m/r9f_<id><mode>/<id>_r9f_<id><mode>.mp4
set -u
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1 METRICS=off
id=$1; shift; A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; export DUB_INPUTS=/workspace/dub/r9/runs/$id/work/in; D=/workspace/dub/r9/diag
for m in "$@"; do
  G=r9g_$id$m; F=r9f_$id$m; t0=$(date +%s); rm -rf $B/$G $B/$F
  AUDIO=$A/r9_$id TAG=$G SKIP_SCORE=1 VIDEO_EXTRA="--codeformer optimized --codeformer-w 1.0 --steps 10 --steps-silent 20 --codeformer-parse off --codeformer-batch 16 --ref-mode $m" \
    bash scripts/r7/r7_e2e.sh $id > $D/ab_${id}_$m.gen.log 2>&1
  t1=$(date +%s)
  SKIP_SCORE=1 TAG=$F SRC_TAG=$G CFD=self SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$(cat reports/experiments/r8-face/fb7.args)" bash scripts/r7/r7_m7_comp.sh $id > $D/ab_${id}_$m.comp.log 2>&1
  t2=$(date +%s); echo "== $m gen $((t1 - t0)) s comp $((t2 - t1)) s"
  [ -f $B/$F/${id}_$F.mp4 ] || { echo "FAIL $m"; continue; }
  ffmpeg -loglevel error -y -i $B/$F/${id}_$F.mp4 -r 25 -an -c:v libx264 -crf 12 /tmp/ab_$m.mp4
  flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video /tmp/ab_$m.mp4 --audio $A/r9_$id/$id/dubbed_16k.wav --out $D/ab_${id}_$m.sync.json > /dev/null 2>&1
  python3 - $D/ab_${id}_$m.sync.json <<'PY'
import json, sys
j = json.load(open(sys.argv[1])); ps = [x for x in j["per_second"] if (x.get("dub_act") or 0) >= 0.5 and x.get("conf_at0") is not None]
print("   tracks", [(t["conf"], t["offset"]) for t in j["tracks"]], "| sync_speech (mean conf@0 over dub-speech seconds)", round(sum(x["conf_at0"] for x in ps) / max(1, len(ps)), 2), "n", len(ps))
PY
done
