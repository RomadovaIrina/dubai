#!/usr/bin/env python3
"""r15 Z: the worm-task table: per video and arm, alone and + agent F's colour lock (<arm>lk): SyncNet (eval/<arm>_<id>.sync.txt), F's
lip_flicker excess blinks / 10 s and the steady-shape dL (eval/lipflick.jsonl, the last line per label), face GPU s / vmin
(<arm>_<id>/timing.json; the lk variant = the same generator).
    python scripts/r15/ztable.py [--arms q2d q2as q2c8 te2 te2k q2s4] [--ids m18 old03 m03]"""
import argparse, json, pathlib, re
X = pathlib.Path("/workspace/dub/r15w/eval")
ap = argparse.ArgumentParser(); ap.add_argument("--arms", nargs="+", default="q2d q2as q2c8 te2 te2k q2s4".split()); ap.add_argument("--ids", nargs="+", default=["m18", "old03", "m03"])
a = ap.parse_args()
LF = {}
for l in (X / "lipflick.jsonl").read_text().splitlines() if (X / "lipflick.jsonl").exists() else []:
    try: d = json.loads(l); LF[d["label"]] = d
    except Exception: pass


def sync(n):
    p = X / f"{n}.sync.txt"; m = re.search(r":\s*([-0-9.]+) over", p.read_text()) if p.exists() else None
    return m.group(1) if m else "–"


def gpu(arm, i):
    try: return json.loads((X / f"{arm}_{i}" / "timing.json").read_text())["total"]["gpu_busy_s_per_vmin"]
    except Exception: return "–"


print("| video | arm | SyncNet | excess blinks / 10 s | dL steady | + colour lock: SyncNet | + colour lock: excess blinks / 10 s | face GPU s / vmin |")
print("|---|---|---|---|---|---|---|---|")
for i in a.ids:
    for arm in a.arms:
        f0, f1 = LF.get(f"{arm}_{i}", {}), LF.get(f"{arm}lk_{i}", {})
        if not f0 and sync(f"{arm}_{i}") == "–": continue
        print(f"| {i} | {arm} | {sync(f'{arm}_{i}')} | {f0.get('excess_per_10s', '–')} | {f0.get('dL_steady', '–')} | {sync(f'{arm}lk_{i}')} | {f1.get('excess_per_10s', '–')} | {gpu(arm, i)} |")
