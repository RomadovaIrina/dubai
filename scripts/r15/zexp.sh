#!/bin/bash
# r15 Z: agent F's explode.py (mouth-shape SNAPS / pops: a frame unlike both flow-warped neighbours, excess over the source > 5 grey levels)
# on a W arm final; all events (--top 1000), the layer after CodeFormer (ls_raw) as the stage of origin -> eval/<arm>_<id>.explode.json + explode.jsonl
#   bash scripts/r15/zexp.sh <id> <arm> [<generator arm, default = arm>]
set -u; id=$1; arm=$2; gen=${3:-$2}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."; . scripts/env.sh >/dev/null
X=/workspace/dub/r15w/eval; B=/workspace/dub/artifacts/r4m; ID=q2_$id; GT=r9g_${ID}_$gen; W=$B/$GT/work_${ID}_$GT
V=$X/${arm}_$id/${arm}_$id.mp4; SRC=$(ls $X/$ID/e2e/work/in/$ID.* | head -1); J=$X/${arm}_$id.explode.json
python scripts/r15/explode.py --work $W --timing $B/$GT/timing/${ID}_master_timing_F.json --src $SRC --src-lm $B/$GT/timing/${ID}_source_lm_F.npz \
  --final ${arm}_$id=$V --stage raw=$W/${ID}_${GT}_25fps_ls_raw.mp4 --top 1000 --json $J > $X/${arm}_$id.explode.log 2>&1
python3 - "$X" "${arm}_$id" <<'PY'
import json, re, sys, pathlib
X, lab = pathlib.Path(sys.argv[1]), sys.argv[2]; log = (X / f"{lab}.explode.log").read_text()
m = re.search(rf"{re.escape(lab)}: pop mean ([0-9.]+), excess mean ([-0-9.]+), frames > thr (\d+)", log); s = re.search(r"source pop mean ([0-9.]+)", log)
try: r = json.loads((X / f"{lab}.explode.json").read_text())
except Exception: r = {"events": [], "origin_counts": {}}
row = {"label": lab, "src_pop": float(s.group(1)) if s else None, "pop": float(m.group(1)) if m else None, "excess": float(m.group(2)) if m else None,
       "frames_gt_thr": int(m.group(3)) if m else None, "events": len(r["events"]), "strong_events_gt10": sum(1 for e in r["events"] if max(e["final_excess"].values()) > 10),
       "origin": r["origin_counts"]}
open(X / "explode.jsonl", "a").write(json.dumps(row) + "\n"); print("== explode", json.dumps(row))
PY
