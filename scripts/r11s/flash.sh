#!/bin/bash
# r11 SYNC: the gate's strong-flash count (r3_flash.py, mc_ratio >= 2 and mc_vs_src >= 2, as r4_gate) of arms, fast (no full gate).
#   bash scripts/r11s/flash.sh <id> <tag> [tag ...]   -> /workspace/dub/r11s/flash/<id>_<tag>.json + one line each
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; id=$1; shift; B=/workspace/dub/artifacts/r4m; P=$B/r9g_$id/work_${id}_r9g_$id/${id}_r9g_${id}_25fps
SRC=$(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1); LM=$B/r9g_$id/timing/${id}_source_lm_F.npz; mkdir -p /workspace/dub/r11s/flash
for t in "$@"; do ( o=/workspace/dub/r11s/flash/${id}_$t.json; [ -s $o ] || python scripts/quality/r3_flash.py --video ${P}_$t.mp4 --src $SRC --lm $LM --out $o > /dev/null 2>&1
  python3 -c "import json,sys;d=json.load(open(sys.argv[1]));s=[e[\"frame\"] for e in d[\"flash_list\"] if (e.get(\"mc_ratio\") or 0)>=2 and (e.get(\"mc_vs_src\") or 0)>=2];print(\"== flash $id $t strong\",len(s),\"all\",d[\"flash_frames\"],s)" $o ) & done; wait
