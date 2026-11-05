#!/usr/bin/env python3
"""r15 F: from PIXELS, the shift of a generated layer against the 25 fps master per frame. Phase correlation of the upper face (eyes / nose box
from the master landmarks: outside the generated mouth, so it must coincide) and of the lower face (the pasted region), in IOD.
A displaced LatentSync paste-back (the m03 19 s 'slid / lopsided mouth') shows as a large lower-face shift with a small upper one or both large.
    python scripts/r15/layer_shift.py <run> [layer suffix, default ls_raw] [--out shifts_<run>.npz]"""
import sys, pathlib, json
import numpy as np, cv2
run = sys.argv[1]; suf = sys.argv[2] if len(sys.argv) > 2 else "ls_raw"
B = "/workspace/dub/artifacts/r4m"; W = pathlib.Path(f"{B}/r9g_{run}/work_{run}_r9g_{run}"); P = W / f"{run}_r9g_{run}_25fps_{suf}.mp4"
lm = np.load(W / "video/master_mediapipe.npz")["lm"].astype(np.float64)
cm, cl = cv2.VideoCapture(str(W / "video/master_25fps.mp4")), cv2.VideoCapture(str(P)); n = len(lm)
up = np.full((n, 2), np.nan); lo = np.full((n, 2), np.nan); diff = np.full(n, np.nan)
def pc(a, b):
    a, b = np.float32(a), np.float32(b); win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), r = cv2.phaseCorrelate(a, b, win); return dx, dy, r
for k in range(n):
    ok1, fm = cm.read(); ok2, fl = cl.read()
    if not (ok1 and ok2): break
    q = lm[k]
    if not np.isfinite(q[[33, 263, 1, 152, 61, 291]]).all(): continue
    iod = np.linalg.norm(q[33, :2] - q[263, :2]); gm, gl = cv2.cvtColor(fm, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY)
    if np.abs(gm.astype(np.int16) - gl.astype(np.int16)).mean() < 0.5: continue   # not a LatentSync frame (the layer = the master)
    for box, arr in ((((q[33] + q[263]) / 2, 0.9 * iod, 0.55 * iod, -0.15), up), (((q[61] + q[291]) / 2, 0.9 * iod, 0.6 * iod, 0.1), lo)):
        (c, hw, hh, dyc) = box; cx, cy = c[0], c[1] + dyc * iod
        x0, x1, y0, y1 = int(cx - hw), int(cx + hw), int(cy - hh), int(cy + hh)
        if x0 < 0 or y0 < 0 or x1 > gm.shape[1] or y1 > gm.shape[0]: continue
        dx, dy, r = pc(gm[y0:y1, x0:x1], gl[y0:y1, x0:x1]); arr[k] = (dx / iod, dy / iod)
    diff[k] = 1
su, sl = np.linalg.norm(up, axis=1), np.linalg.norm(lo, axis=1)
print(f"== {run} {suf}: LS frames {int(np.nansum(diff))}; upper-face shift p50 {np.nanpercentile(su, 50):.4f} p99 {np.nanpercentile(su, 99):.4f} max {np.nanmax(su):.3f}; "
      f"lower-face shift p50 {np.nanpercentile(sl, 50):.4f} p99 {np.nanpercentile(sl, 99):.4f} max {np.nanmax(sl):.3f} IOD", flush=True)
for thr in (0.03, 0.05):
    bad = np.flatnonzero(np.nan_to_num(np.maximum(su, sl)) > thr); runs, k = [], 0
    while k < len(bad):
        e = k
        while e + 1 < len(bad) and bad[e + 1] <= bad[e] + 2: e += 1
        runs.append([int(bad[k]), int(bad[e]), round(bad[k] / 25, 2), round(float(np.nanmax(np.maximum(su, sl)[bad[k]:bad[e] + 1])), 3)]); k = e + 1
    print(f"   runs over {thr} IOD [b, e, t, max]: {runs}", flush=True)
if run == "q2_m03": print("   m03 468-482 upper / lower:", " ".join(f"{k}:{su[k]:.3f}/{sl[k]:.3f}" for k in range(468, 483)), flush=True)
np.savez_compressed(f"/workspace/dub/r15f/shift_{run}_{suf}.npz", up=up, lo=lo)
