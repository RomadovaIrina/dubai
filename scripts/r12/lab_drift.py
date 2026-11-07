#!/usr/bin/env python3
"""r13 LS quality: colour / tone drift and sharpness of a final vs its source on the same frames (both decoded by the same OpenCV / ffmpeg path, so
the decoder's own matrix cancels; what remains is the drift of our chain).
Regions per sampled frame (source landmarks): bg = outside the face oval dilated 0.6 IOD; box = the lower-face paste box (jaw / cheeks, lips + 0.2
IOD excluded); lips = the outer lip hull. Printed: Lab mean difference (final - source) per region, and the Laplacian-variance ratio
final / source on box and lips (the "мыло" check).
    python scripts/r12/lab_drift.py --video FINAL.mp4 --src SOURCE --src-lm <id>_source_lm_F.npz --label x [--samples 60]"""
import argparse, json
import numpy as np, cv2
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LOW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 1]
ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--src", required=True); ap.add_argument("--src-lm", required=True)
ap.add_argument("--label", default=""); ap.add_argument("--samples", type=int, default=60); a = ap.parse_args()
lm = np.load(a.src_lm)["lm"].astype(np.float64); cv, cs = cv2.VideoCapture(a.video), cv2.VideoCapture(a.src)
n = int(min(cv.get(cv2.CAP_PROP_FRAME_COUNT), cs.get(cv2.CAP_PROP_FRAME_COUNT), len(lm))); pick = set(np.linspace(5, n - 5, a.samples).astype(int).tolist())
acc = {r: [] for r in ("bg", "box", "lips")}; sharp = {r: [] for r in ("box", "lips")}; k = 0
def poly(shape, pts):
    m = np.zeros(shape, np.uint8); cv2.fillPoly(m, [np.round(cv2.convexHull(pts.astype(np.float32))[:, 0]).astype(np.int32)], 1); return m
while k < n:
    okv, fv = cv.read(); oks, fs = cs.read()
    if not (okv and oks): break
    if k in pick:
        p = lm[k]
        if np.isfinite(p[OVAL + LIPS + LOW + [33, 263]]).all():
            iod = float(np.linalg.norm(p[33, :2] - p[263, :2])); H, W = fs.shape[:2]; ker = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(r * iod) + 1,) * 2)
            face = cv2.dilate(poly((H, W), p[OVAL, :2]), ker(0.6)) > 0; lips = poly((H, W), p[LIPS, :2]) > 0
            box = (poly((H, W), p[LOW, :2]) > 0) & ~(cv2.dilate(lips.astype(np.uint8), ker(0.2)) > 0)
            Lv, Ls = cv2.cvtColor(fv, cv2.COLOR_BGR2LAB).astype(np.float32), cv2.cvtColor(fs, cv2.COLOR_BGR2LAB).astype(np.float32)
            Lv[..., 0] *= 100 / 255; Ls[..., 0] *= 100 / 255; Lv[..., 1:] -= 128; Ls[..., 1:] -= 128
            for r, m in (("bg", ~face), ("box", box), ("lips", lips)):
                if m.sum() > 50: acc[r].append((Lv[m] - Ls[m]).mean(0))
            gv, gs = cv2.cvtColor(fv, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.cvtColor(fs, cv2.COLOR_BGR2GRAY).astype(np.float32)
            lv, ls = cv2.Laplacian(gv, cv2.CV_32F), cv2.Laplacian(gs, cv2.CV_32F)
            for r, m in (("box", box), ("lips", lips)):
                if m.sum() > 50: sharp[r].append(float(lv[m].var() / max(ls[m].var(), 1e-3)))
    k += 1
res = {"label": a.label}
for r, v in acc.items():
    if v: v = np.array(v); res[r] = {"dL": round(float(v[:, 0].mean()), 2), "da": round(float(v[:, 1].mean()), 2), "db": round(float(v[:, 2].mean()), 2)}
for r, v in sharp.items():
    if v: res["sharp_" + r] = round(float(np.median(v)), 3)
print("== lab_drift", json.dumps(res), flush=True)
