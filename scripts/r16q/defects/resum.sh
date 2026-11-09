#!/bin/bash
# re-summarise every scored video from its saved series (no decoding)
Q=/workspace/dub/q_tmp; cd $Q/defects
for j in *.json; do case $j in *.det.json|CALIB*) continue;; esac; id=${j%.json}; [ -s $id.series.npz ] || continue
  S=$(python3 -c "import json;print(json.load(open(\"$j\"))[\"src\"])"); O=$(python3 -c "import json;print(json.load(open(\"$j\"))[\"ours\"])")
  /venv/mediapipe/bin/python $Q/scripts/r16q_defects.py --src "$S" --ours "$O" --out $j --npz $id.det.npz --label $id --from-series $id.series.npz > $id.resum.log 2>&1 || echo "FAIL $id"
done; python3 $Q/scripts/defects_calib.py . | head -3
