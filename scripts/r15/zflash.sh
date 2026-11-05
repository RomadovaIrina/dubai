#!/bin/bash
# r15 Z: the QA gate's flash detector (scripts/quality/r3_flash.py as r3_qa_gates runs it) on a W arm final -> eval/<arm>_<id>.flash.json,
# summary -> eval/flash.jsonl: flash_frames, flash_mc, flash_stat and STRONG = mc_ratio >= 2 and mc_vs_src >= 2 (the gate's strong-flash list,
# agent F's m03 "snaps": open -> half -> pursed within 2 frames).      bash scripts/r15/zflash.sh <id> <arm> [<generator arm>]
set -u; id=$1; arm=$2; gen=${3:-$2}
cd "$(dirname "${BASH_SOURCE[0]}")/../.."; . scripts/env.sh >/dev/null
X=/workspace/dub/r15w/eval; B=/workspace/dub/artifacts/r4m; ID=q2_$id; GT=r9g_${ID}_$gen
V=$X/${arm}_$id/${arm}_$id.mp4; SRC=$(ls $X/$ID/e2e/work/in/$ID.* | head -1); J=$X/${arm}_$id.flash.json
python scripts/quality/r3_flash.py --video $V --src $SRC --lm $B/$GT/timing/${ID}_source_lm_F.npz --out $J > $X/${arm}_$id.flash.log 2>&1
python3 - "$J" "${arm}_$id" "$X/flash.jsonl" <<'PY'
import json, sys
J, lab, out = sys.argv[1:4]
try: d = json.load(open(J))
except Exception as e: print("== flash", lab, "FAILED", e); sys.exit(0)
fl = d.get("flash_list", []); st = [e for e in fl if (e.get("mc_ratio") or 0) >= 2.0 and (e.get("mc_vs_src") or 0) >= 2.0]
row = {"label": lab, "frames": d.get("frames"), "flash_frames": d.get("flash_frames"), "flash_mc": d.get("flash_mc"), "flash_stat": d.get("flash_stat"),
       "strong": len(st), "strong_t": [e["t"] for e in st][:40]}
open(out, "a").write(json.dumps(row) + "\n"); print("== flash", json.dumps({k: row[k] for k in ("label", "flash_frames", "flash_mc", "strong")}))
PY
