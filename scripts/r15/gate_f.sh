#!/bin/bash
# r15 F: scripts/r10/gate.sh for the r14-d layout (agent D's q2 eval runs, inputs under /workspace/dub/r14d/eval/<run>/e2e/work/in), run from this
# checkout, output under /workspace/dub/r15f/gate/<run><sfx>/. The video must already be artifacts/r4m/r9f_<run><sfx>/<run>_r9f_<run><sfx>.mp4 (comp_f.sh).
#   bash scripts/r15/gate_f.sh <run, e.g. q2_m03> <sfx>
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd $R; . scripts/env.sh >/dev/null; export DUB_ROOT=$R DUB_ALLOW_NC=1 METRICS=on
id=$1; sx=$2; B=/workspace/dub/artifacts/r4m; A=/workspace/dub/artifacts/r4a; GT=r9g_$id; FT=r9f_$id$sx; AT=r9_$id
export DUB_INPUTS=/workspace/dub/r14d/eval/$id/e2e/work/in; SRC=$(ls $DUB_INPUTS/$id.* | head -1)
OUT=/workspace/dub/r15f/gate/$id$sx; mkdir -p $OUT/logs /workspace/dub/r9/diag
ln -sfn $B/$GT/timing $B/$FT/timing 2>/dev/null
[ -f $B/$FT/${id}_$FT.manifest.json ] || python - $B/r9f_$id/${id}_r9f_$id.manifest.json $B/$FT/${id}_$FT.manifest.json $B/$FT/${id}_$FT.mp4 <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); m["output"] = sys.argv[3]; json.dump(m, open(sys.argv[2], "w"), indent=1)
PY
cp /workspace/dub/r14d/eval/$id/e2e/timing.json $OUT/timing.json 2>/dev/null
python scripts/r4/r4_gate.py score --label $FT --dir $B/$FT --ids $id --force > $OUT/logs/face_score.log 2>&1 || echo "WARN face score"
python scripts/r7/r7_closed_mouth.py --ids $id --variants $FT:$A/$AT --out $OUT/closed_mouth.json > /dev/null 2>&1
python scripts/r9/occl_scan.py --src $SRC --out $B/$FT/${id}_$FT.mp4 --lm $B/$GT/timing/${id}_source_lm_F.npz --sheet $OUT/occl.jpg --json $OUT/occlusion.json 2>&1 | tail -n 1
CUDA_VISIBLE_DEVICES= python scripts/r9/artefacts.py $id --tag-suffix "$sx" --gen $GT --out $OUT/art.json 2>&1 | grep -v Warn | tail -1
bash scripts/r7/r7_frame_gate.sh $B/$FT:$FT $id | grep "^=="
python scripts/r7/r7_baseline.py freeze --tag $FT --audio $A/$AT --ids $id --out $OUT/face_gate.json | tail -1
echo "R15_GATE_DONE $id$sx"
