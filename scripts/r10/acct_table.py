#!/usr/bin/env python3
"""r10: config -> per-stage wall / GPU-busy s per video-minute from e2e run dirs (gpu_time.json + timing.json).
    python scripts/r10/acct_table.py <label>=<run dir> ...      (markdown table on stdout)"""
import json, pathlib, sys
ST = ("audio", "face_gen", "face_comp")
print("| run | video min | " + " | ".join(f"{s} wall / GPU" for s in ST) + " | production wall | **production GPU-busy** | GPU util-weighted |")
print("|---|---:|" + "---:|" * (len(ST) + 3))
for arg in sys.argv[1:]:
    lab, d = arg.split("=", 1); d = pathlib.Path(d)
    g = json.loads((d / "gpu_time.json").read_text()); t = json.loads((d / "timing.json").read_text()); s = g["stages"]; vm = t["video_min"]
    prod = {k: v for k, v in s.items() if k not in ("metrics", "wait_heavy", "wait_comp")}
    w = sum(v["wall_s"] for v in prod.values()) / vm; b = sum(v["gpu_busy_s"] for v in prod.values()) / vm; u = sum(v["gpu_weighted_s"] for v in prod.values()) / vm
    cell = lambda k: f"{s[k]['wall_s_per_vmin']:.0f} / {s[k]['gpu_busy_s_per_vmin']:.0f}" if k in s else "-"
    print(f"| {lab} | {vm:.3f} | " + " | ".join(cell(k) for k in ST) + f" | {w:.0f} | **{b:.0f}** | {u:.0f} |")
