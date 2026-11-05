#!/usr/bin/env python3
"""r15 F: 'jaw drift' of a generated layer vs the 25 fps master, per frame. LatentSync must only redraw the mouth; on m03 19 s it redrew the
whole lower face moved / rotated (hair entering at the left, the mouth pushed down-left) while the source does not move. Score = mean
|gradient magnitude(layer) - gradient magnitude(master)| over the lower-face ring (nose tip down to the chin, inside the jaw line) OUTSIDE the
lips dilated by 0.25 IOD, over the same for the video's median frame -> 1 = typical, >> 1 = the generator moved the jaw / cheeks.
    python scripts/r15/jaw_drift.py <run> [layer suffix, default ls_raw] [--thr 2.5]"""
import sys, pathlib, json, argparse
import numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("suf", nargs="?", default="ls_raw"); ap.add_argument("--thr", type=float, default=2.5)
a = ap.parse_args(); B = "/workspace/dub/artifacts/r4m"; W = pathlib.Path(f"{B}/r9g_{a.run}/work_{a.run}_r9g_{a.run}")
lm = np.load(W / "video/master_mediapipe.npz")["lm"].astype(np.float64); n = len(lm)
JAW = [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361]; LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
cm, cl = cv2.VideoCapture(str(W / "video/master_25fps.mp4")), cv2.VideoCapture(str(W / f"{a.run}_r9g_{a.run}_25fps_{a.suf}.mp4")); s = np.full(n, np.nan)
def gmag(g): return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
for k in range(n):
    ok1, fm = cm.read(); ok2, fl = cl.read()
    if not (ok1 and ok2): break
    q = lm[k]
    if not np.isfinite(q[JAW + LIPS + [1, 33, 263]]).all(): continue
    iod = np.linalg.norm(q[33, :2] - q[263, :2]); poly = np.vstack([q[[234]][:, :2], q[JAW, :2], q[[454]][:, :2]])
    x0, y0 = np.floor(poly.min(0) - 4).astype(int); x1, y1 = np.ceil(poly.max(0) + 4).astype(int); y0 = int(q[1, 1])   # from the nose tip down
    H, Wd = fm.shape[:2]; x0, y0, x1, y1 = max(0, x0), max(0, y0), min(Wd, x1), min(H, y1)
    if x1 - x0 < 16 or y1 - y0 < 16: continue
    m = np.zeros((y1 - y0, x1 - x0), np.uint8); cv2.fillPoly(m, [np.int32(np.vstack([q[[234]][:, :2], q[JAW, :2], q[[454]][:, :2]]) - [x0, y0])], 1)
    lip = np.zeros_like(m); cv2.fillPoly(lip, [np.int32(q[LIPS, :2] - [x0, y0])], 1); r = max(1, int(0.25 * iod)); lip = cv2.dilate(lip, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    ring = (m > 0) & (lip == 0)
    if ring.sum() < 50: continue
    gm = gmag(cv2.cvtColor(fm[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)); gl = gmag(cv2.cvtColor(fl[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32))
    s[k] = float(np.abs(gl - gm)[ring].mean())
med = np.nanmedian(s); z = s / med; bad = np.flatnonzero(np.nan_to_num(z) > a.thr); runs, k = [], 0
while k < len(bad):
    e = k
    while e + 1 < len(bad) and bad[e + 1] <= bad[e] + 2: e += 1
    runs.append([int(bad[k]), int(bad[e]), round(bad[k] / 25, 2), round(float(np.nanmax(z[bad[k]:bad[e] + 1])), 2)]); k = e + 1
print(f"== {a.run} {a.suf}: jaw drift (x median) p50 1.00 p95 {np.nanpercentile(z, 95):.2f} p99 {np.nanpercentile(z, 99):.2f} max {np.nanmax(z):.2f}; runs > {a.thr}: {runs}", flush=True)
if a.run == "q2_m03": print("   m03 466-484:", " ".join(f"{k}:{z[k]:.2f}" for k in range(466, 485)), flush=True)
np.save(f"/workspace/dub/r15f/jaw_{a.run}_{a.suf}.npy", z)
