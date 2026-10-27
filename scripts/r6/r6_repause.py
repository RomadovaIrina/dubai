#!/usr/bin/env python3
"""r6: recompute the pause-logic axis (r4_pause_logic, current PICK) of finished runs in place (<tag>/<id>.metrics.json: pause_logic + pause_rows),
so every tag of a gate is scored by the same classifier without re-running the whole metrics stage.
    python scripts/r6/r6_repause.py <tag> [tag ...]"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "quality"))
import r4_pause_logic as pl
A = pathlib.Path("/workspace/dub/artifacts/r4a")
for tag in sys.argv[1:]:
    for id_ in ["01", "02", "03", "04", "05"]:
        f = A / tag / f"{id_}.metrics.json"; run = A / tag / id_; mp = run / f"{id_}_r4b.manifest.json"
        if not (f.exists() and mp.exists()): continue
        m = json.loads(f.read_text()); r = pl.analyse(run, json.loads(mp.read_text()))
        m["pause_logic"] = {k: v for k, v in r.items() if k != "rows"}; m["pause_logic"]["pick"] = pl.PICK; m["pause_rows"] = r["rows"]
        f.write_text(json.dumps(m, indent=1, ensure_ascii=False)); print(tag, id_, r["pauses_inside_phrase"], r["inside_phrase_s"], flush=True)
