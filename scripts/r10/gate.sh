#!/bin/bash
# r10 FACE: the r9-speed-face gate rows (face_gate.json: SyncNet / flashes / pressed lips / frame gate; art.json: stick / slide; occlusion.json)
# for ANY face video of a baseline id, laid out for scripts/r9/gate_cmp.py (git show origin/experiment/r9-speed-face:scripts/r9/gate_cmp.py).
#   bash scripts/r10/gate.sh <id> <suffix> [video]   (suffix "" + no video = the r9 + fix baseline itself) -> /workspace/dub/r10/gate/<id><suffix|base>/
set -u
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1 METRICS=on
id=$1; sx=$2; V=${3:-}; B=/workspace/dub/artifacts/r4m; A=/workspace/dub/artifacts/r4a; GT=r9g_$id; FT=r9f_$id$sx; AT=r9_$id
export DUB_INPUTS=/workspace/dub/r10/runs/$id/work/in; SRC=$(ls $DUB_INPUTS/$id.* | head -1)
OUT=/workspace/dub/r10/gate/$id${sx:-base}; mkdir -p $OUT/logs $B/$FT /workspace/dub/r9/diag
if [ -n "$V" ]; then   # candidate: the video + the baseline composite's manifest (same run, same audio)
  ln -sfn $B/$GT/timing $B/$FT/timing 2>/dev/null
  ffmpeg -loglevel error -y -i $V -i $B/r9f_$id/${id}_r9f_$id.mp4 -map 0:v:0 -map 1:a:0 -c copy $B/$FT/${id}_$FT.mp4
  python - $B/r9f_$id/${id}_r9f_$id.manifest.json $B/$FT/${id}_$FT.manifest.json $B/$FT/${id}_$FT.mp4 <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); m["output"] = sys.argv[3]; json.dump(m, open(sys.argv[2], "w"), indent=1)
PY
fi
cp /workspace/dub/r10/runs/$id/timing.json $OUT/timing.json 2>/dev/null
python scripts/r4/r4_gate.py score --label $FT --dir $B/$FT --ids $id --force > $OUT/logs/face_score.log 2>&1 || echo "WARN face score"
python scripts/r7/r7_closed_mouth.py --ids $id --variants $FT:$A/$AT --out $OUT/closed_mouth.json > /dev/null 2>&1
python scripts/r9/occl_scan.py --src $SRC --out $B/$FT/${id}_$FT.mp4 --lm $B/$GT/timing/${id}_source_lm_F.npz --sheet $OUT/occl.jpg --json $OUT/occlusion.json 2>&1 | tail -n 1
CUDA_VISIBLE_DEVICES= python scripts/r9/artefacts.py $id --tag-suffix "$sx" --gen $GT --out $OUT/art.json 2>&1 | grep -v Warn | tail -1
bash scripts/r7/r7_frame_gate.sh $B/$FT:$FT $id | grep "^=="
python scripts/r7/r7_baseline.py freeze --tag $FT --audio $A/$AT --ids $id --out $OUT/face_gate.json | tail -1
echo "R10_GATE_DONE $id$sx"
