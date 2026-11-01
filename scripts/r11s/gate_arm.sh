#!/bin/bash
# r11 SYNC: the r9-speed-face gate rows of an arm (scripts/r10/gate.sh, run from the main checkout as the face agent does) + gate_cmp vs <id>base.
#   bash scripts/r11s/gate_arm.sh <id> <tag>   -> /workspace/dub/r10/gate/<id>Z<tag>/, row appended to /workspace/dub/r11s/gate.md
id=$1; T=$2; P=/workspace/dub/artifacts/r4m/r9g_$id/work_${id}_r9g_$id/${id}_r9g_${id}_25fps; R=/workspace/dub/r10/gate
[ -s $R/${id}base/face_gate.json ] || bash /workspace/dub/dubai/scripts/r10/gate.sh $id "" > /workspace/dub/r11s/logs/gate_${id}base.log 2>&1
bash /workspace/dub/dubai/scripts/r10/gate.sh $id Z$T ${P}_$T.mp4 > /workspace/dub/r11s/logs/gate_${id}_$T.log 2>&1
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null
python scripts/r10/gate_cmp.py --base $R/${id}base --cand $R/${id}Z$T --md /workspace/dub/r11s/gate.md | head -n 3
