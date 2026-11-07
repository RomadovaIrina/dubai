#!/usr/bin/env python3
"""r16 L: collect one eval dir's per-tag metrics (flow / landmark cheek, SyncNet, cavity) into metrics.json.   python l_agg.py <eval dir>"""
import json, sys, glob, os

O = sys.argv[1]; res = {}
ld = lambda p: json.load(open(p)) if os.path.exists(p) else None
for f in sorted(glob.glob(f"{O}/*.cheek.json")):
    t = os.path.basename(f).split(".")[0]; c = ld(f); r = {"cheek_lm": c["ratio"]}
    fl = ld(f"{O}/{t}.flow.json"); r["flow"] = fl
    s = ld(f"{O}/{t}.sync.json"); r["sync"] = {"conf": s["global"]["conf"], "off": s["global"]["av_offset"]} if s else None
    cv = ld(f"{O}/{t}.cav.json")
    if cv: r["cavity"] = {k: {kk: v.get(kk) for kk in ("open_frames", "black_hole_frames", "black_hole_share", "near_black_share", "black_hole_times")} for k, v in cv.items()}
    b = ld(f"{O}/{t}.bnd.json"); r["boundary13"] = {f"ph{ph}": {k: v["ratio"] for k, v in b[f"P13_ph{ph}"].items()} for ph in (0, 1)} if b and "P13_ph0" in b else None
    res[t] = r
json.dump(res, open(f"{O}/metrics.json", "w"), indent=1)
for t, r in res.items():
    print(os.path.basename(O), t, "flow", (r["flow"] or {}).get("ratio"), "spike", (r["flow"] or {}).get("spike"), "sync", r["sync"], "bnd13", r.get("boundary13"),
          "black", {k: (v["black_hole_frames"], v["open_frames"], v["near_black_share"]) for k, v in (r.get("cavity") or {}).items()})
