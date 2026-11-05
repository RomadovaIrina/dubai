#!/usr/bin/env python3
"""r15 W1: best-of-N pick statistics per arm / video from eval/<arm>_<id>.bon.jsonl (one line per generator segment, scripts/r15/bon.py):
windows, share where the pick != seed 0, the pick histogram, generation + scoring seconds, and how often each score term decided the pick
(the term with the largest weighted spread between the pick and seed 0).
    python scripts/r15/bon_stats.py"""
import glob, json, pathlib
import numpy as np
W = np.array([1.0, 0.5, 0.5, -0.5]) / np.array([0.03, 0.05, 0.15, 0.30]); TERMS = ["sync", "open", "sharp", "jump"]
for f in sorted(glob.glob("/workspace/dub/r15w/eval/*.bon.jsonl")):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    if not rows: continue
    win = sum(r["windows"] for r in rows); ns = sum(r["not_seed0"] for r in rows); N = rows[0]["N"]
    hist = np.sum([r["pick_hist"] for r in rows], 0).tolist(); sec = sum(r["seconds"] for r in rows); sco = sum(r["score_s"] for r in rows)
    dec = {t: 0 for t in TERMS}
    for r in rows:
        for w in r["rows"]:
            if w["pick"] == 0: continue
            raw = np.array(w["raw"], float); d = (raw[w["pick"]] - raw[0]) * W; d[~np.isfinite(d)] = 0
            dec[TERMS[int(np.argmax(d))]] += 1
    print(f"{pathlib.Path(f).stem.replace('.bon', ''):14s} N={N}{' early' if rows[0].get('early') else '':6s} windows {win:4d}  pick != seed0 {ns:4d} ({100 * ns / max(1, win):.0f} %)  "
          f"hist {hist}  {sec:.0f} s (scoring {sco:.0f} s)  deciding term {dec}")
