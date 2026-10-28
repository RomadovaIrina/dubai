#!/usr/bin/env python3
"""r7 F: sharpness of the pasted mouth region BY DISTANCE FROM THE LIPS (user: "blurred / soapy lower face"). Per sampled frame, on the
changed pixels C (blur(|out - src|) > 6), the ratio mean |Laplacian out| / mean |Laplacian src| in bands of the distance to the source lips
hull (in mouth widths): lips, 0-0.15, 0.15-0.3, 0.3-0.5, 0.5-0.8; plus 'control' = the same ratio on UNCHANGED pixels of the lower face
(encode loss only: the ceiling). Median over the frames. 1.0 = as crisp as the source.
    python scripts/r7/r7_sharp_probe.py --src SRC --lm <src mediapipe npz> --videos LABEL=PATH [LABEL=PATH ...] [--every 60] [--json out.json]
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np

LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
BANDS = [("lips", -1, 0), ("0-0.15", 0, 0.15), ("0.15-0.3", 0.15, 0.3), ("0.3-0.5", 0.3, 0.5), ("0.5-0.8", 0.5, 0.8)]


def grab(path, ks):
    import av
    c = av.open(path); out = {}
    for k, fr in enumerate(c.decode(c.streams.video[0])):
        if k in ks: out[k] = fr.to_ndarray(format="gray").astype(np.float32)
        if k >= max(ks): break
    c.close(); return out


def main():
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--every", type=int, default=60); ap.add_argument("--start", type=int, default=200); ap.add_argument("--json", default=None)
    a = ap.parse_args(); lm = np.load(a.lm)["lm"]; ks = set(range(a.start, len(lm) - 1, a.every)); S = grab(a.src, ks); rep = {}
    for item in a.videos:
        name, path = item.split("=", 1); O = grab(path, ks); res = {b[0]: [] for b in BANDS}; res["control"] = []; chg = []
        for k in sorted(ks):
            if k not in O or k not in S: continue
            p = lm[k]
            if not np.isfinite(p[LIPS]).all(): continue
            q = p[LIPS, :2]; mw = q[:, 0].max() - q[:, 0].min(); hull = cv2.convexHull(q.astype(np.float32))[:, 0]
            m = np.zeros(S[k].shape, np.uint8); cv2.fillPoly(m, [np.round(hull).astype(np.int32)], 1); dist = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
            C = cv2.GaussianBlur(np.abs(O[k] - S[k]), (0, 0), 2) > 6
            ls = np.abs(cv2.Laplacian(S[k], cv2.CV_32F, ksize=3)); lo = np.abs(cv2.Laplacian(O[k], cv2.CV_32F, ksize=3))
            for b, lo_, hi_ in BANDS:
                sel = ((dist > lo_ * mw) & (dist <= hi_ * mw) & C) if lo_ >= 0 else ((m > 0) & C)
                if sel.sum() > 50: res[b].append(lo[sel].mean() / max(ls[sel].mean(), 1e-3))
            ctl = (dist > 0.8 * mw) & (dist < 2.0 * mw) & ~C
            if ctl.sum() > 200: res["control"].append(lo[ctl].mean() / max(ls[ctl].mean(), 1e-3))
            chg.append(int(C.sum()))
        rep[name] = {b: (round(float(np.median(v)), 3) if v else None) for b, v in res.items()}; rep[name]["n"] = {b: len(v) for b, v in res.items()}; rep[name]["changed_px_median"] = int(np.median(chg)) if chg else 0
        print(f"== sharp {name:10s} " + " ".join(f"{b}:{rep[name][b]}" for b in [x[0] for x in BANDS] + ["control"]) + f"  changed px {rep[name]['changed_px_median']}")
    if a.json: json.dump(rep, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
