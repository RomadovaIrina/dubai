#!/usr/bin/env python3
"""r16 Q: per-detector DIFF of an e2e final vs the eye-accepted final of the same clip (both run by r16q_detect.py against the SAME master).
Reports flag frames e2e / accepted per detector and the e2e-only flag ranges (e2e flagged where the accepted final has no flag within +-2 frames).
    python3 det_diff.py --e2e det_e2e_<id>.json --acc det_acc_<id>.json [--label id]"""
import argparse, json
import numpy as np
from scipy.ndimage import binary_dilation
ap = argparse.ArgumentParser(); ap.add_argument("--e2e", required=True); ap.add_argument("--acc", required=True); ap.add_argument("--label", default=""); a = ap.parse_args()
E, A = json.load(open(a.e2e)), json.load(open(a.acc))
FPS = 25.0
n = 1
for R in (E, A):
    for v in R["det"].values():
        for r in v["ranges"]: n = max(n, int(round(r[1] * FPS)) + 3)


def mask(R, name):
    m = np.zeros(n, bool)
    for r in R["det"][name]["ranges"]: m[int(round(r[0] * FPS)):int(round(r[1] * FPS)) + 1] = True
    return m


out = {"label": a.label, "det": {}}
for name in E["det"]:
    me, ma = mask(E, name), mask(A, name)
    only = me & ~binary_dilation(ma, iterations=2)
    runs, i = [], 0
    while i < n:
        if only[i]:
            j = i
            while j + 1 < n and only[j + 1]: j += 1
            runs.append([round(i / FPS, 2), round(j / FPS, 2)]); i = j + 1
        else: i += 1
    out["det"][name] = {"e2e": E["det"][name]["flag_frames"], "acc": A["det"][name]["flag_frames"], "diff": E["det"][name]["flag_frames"] - A["det"][name]["flag_frames"],
                        "e2e_only_frames": int(only.sum()), "e2e_only_ranges": runs[:15]}
print(json.dumps(out))
