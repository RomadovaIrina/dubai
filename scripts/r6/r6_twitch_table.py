#!/usr/bin/env python3
"""Round 6 F: one markdown table over the twitch scores of several versions (r6_twitch.sh labels) + the top events per video.
    python scripts/r6/r6_twitch_table.py v4 r4_video_v1 m3 [--top 5] [--root /workspace/dub/artifacts/r6f/twitch]"""
import argparse, json, pathlib
ap = argparse.ArgumentParser(); ap.add_argument("labels", nargs="+"); ap.add_argument("--top", type=int, default=5)
ap.add_argument("--root", default="/workspace/dub/artifacts/r6f/twitch"); ap.add_argument("--ids", default="01 02 03 04 05")
a = ap.parse_args(); root = pathlib.Path(a.root); ids = a.ids.split()
load = lambda lab, i: json.loads((root / lab / f"{i}.twitch.json").read_text()) if (root / lab / f"{i}.twitch.json").exists() else None
for key in ("twitch_sum", "n_events", "n_events_ge2", "top20_mean", "p99"):
    print(f"\n**{key}** (lower = better)\n\n| video | " + " | ".join(a.labels) + " |\n|---|" + "---|" * len(a.labels))
    for i in ids:
        row = []
        for lab in a.labels:
            d = load(lab, i); row.append("-" if d is None else str(d["summary"][key]))
        print(f"| {i} | " + " | ".join(row) + " |")
print("\n**top events** (t s: score channel cause head-speed)")
for lab in a.labels:
    for i in ids:
        d = load(lab, i)
        if d is None: continue
        ev = sorted(d["events"], key=lambda e: -e["score"])[:a.top]
        print(f"- {lab} {i}: " + "; ".join(f"{e['t']:.1f}s {e['score']:.1f} {e['channel']} {e.get('cause', '-')} h{e.get('head_speed_iod_s')}" for e in ev))
