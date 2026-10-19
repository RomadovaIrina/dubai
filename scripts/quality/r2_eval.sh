#!/bin/bash
# Round 2 (track A): SyncNet (dub-only + final mix) and the common metric v4 (mouth_jitter_hr.py) on window arms.
#   r2_eval.sh <id> <t0> <t1> <wtag> <arm>...
set -u
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai
id=$1; t0=$2; t1=$3; w=$4; shift 4
O=/workspace/dub/artifacts/lipsync-temporal/r2/win; M=/workspace/dub/artifacts/lipsync-temporal/r2/masters
f0=$(python3 -c "print(int(round($t0*25)))"); f1=$(python3 -c "print(int(round($t1*25)))"); n=$((f1-f0))
REF=$O/${id}_${w}_ref.mp4
[ -f $REF ] || ffmpeg -loglevel error -y -nostdin -i $M/$id/master_25fps.mp4 -vf "select=between(n\,$f0\,$((f1-1)))" -vsync 0 -c:v libx264 -crf 0 -an $REF
for arm in "$@"; do
  tag=${w}_$arm; V=$O/${id}_$tag.mp4
  [ -f $V ] || { echo "missing $V"; continue; }
  if [ ! -f $O/${id}_$tag.sync.json ]; then
    /venv/dabai/bin/python - "$O/${id}_${tag}_dubonly.mp4" "$V" "$O/${id}_$tag.sync.json" <<'PY'
import sys, json, pathlib; sys.path.insert(0, "scripts/pilot")
from syncnet_common import eval_syncnet
r = {}
for k, p in (("dub_only", sys.argv[1]), ("mix", sys.argv[2])):
    try: x = eval_syncnet(pathlib.Path(p)); r[k] = {"conf": x["confidence"], "av_offset": x["av_offset_frames"]}
    except Exception as e: r[k] = {"error": str(e)[-300:]}
pathlib.Path(sys.argv[3]).write_text(json.dumps(r)); print("sync", sys.argv[2].split("/")[-1], r)
PY
  fi
  if [ ! -f $O/${id}_$tag.v4.json ]; then
    # segments of the window manifest are in absolute master frames -> shift to clip frames for the v4 classes
    /venv/dabai/bin/python -c "
import json,sys; m=json.load(open('$O/${id}_$tag.manifest.json')); f0=$f0
for s in m['lipsync']['segments']: s['start_frame']-=f0; s['end_frame']-=f0
m['lipsync']['master_frames']=$n; json.dump(m,open('$O/${id}_$tag.v4man.json','w'))"
    /venv/dabai/bin/python /workspace/dub/common/mouth_jitter_hr.py --video $V --reference $REF --manifest $O/${id}_$tag.v4man.json --tag $tag --out $O/${id}_$tag.v4.json > $O/${id}_$tag.v4.log 2>&1 || echo "v4 FAIL $tag"
  fi
done
