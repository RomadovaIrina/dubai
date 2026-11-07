#!/usr/bin/env python3
"""r16 L: KeySync segment-boundary jumps. Per frame t: mean |frame_t - frame_t-1| (grey, 512 crop) in the eye region and in the lower-face
band outside the lips (cheek / jaw), plus the band's residual-flow change (l_flow.py style). Folded by period P (13 = 14-frame segments
sharing a keyframe): the phase with the largest mean = the boundary; ratio = boundary mean / other frames mean. Same for the source crop.
    python l_boundary.py --gen G.mp4 --gen-lm G.npz --src S.mp4 --src-lm S.npz [--period 13] [--out B.json]"""
import argparse, json
import numpy as np, cv2

LOW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 352, 346, 347, 348, 349, 329, 371, 266, 425, 411, 427, 207, 187, 205, 36, 142, 101, 118, 117, 116, 123]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
EYES = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 353, 265, 340, 346, 347, 348, 349, 350, 277, 168, 47, 121, 120, 119, 118, 117, 111, 35, 124]


def series(vid, lmf):
    lm = np.load(lmf)["lm"][:, :, :2]; cap = cv2.VideoCapture(vid); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    n = min(len(fr), len(lm)); E, B = [np.nan], [np.nan]
    for t in range(1, n):
        p = lm[t]; iod = np.linalg.norm(p[263] - p[33])
        band = np.zeros((512, 512), np.uint8); cv2.fillPoly(band, [cv2.convexHull(p[LOW].astype(np.int32))], 1)
        lips = np.zeros_like(band); cv2.fillPoly(lips, [cv2.convexHull(p[LIPS].astype(np.int32))], 1)
        k = max(3, int(0.25 * iod)) | 1; lips = cv2.dilate(lips, np.ones((k, k), np.uint8)); band = band.astype(bool) & ~lips.astype(bool)
        eye = np.zeros_like(lips); cv2.fillPoly(eye, [cv2.convexHull(p[EYES].astype(np.int32))], 1); eye = eye.astype(bool) & ~band
        d = np.abs(fr[t] - fr[t - 1]); E.append(float(d[eye].mean())); B.append(float(d[band].mean()))
    return np.array(E), np.array(B)


def fold(x, P):
    ok = np.isfinite(x); idx = np.arange(len(x)); m = [float(np.nanmean(x[ok & (idx % P == k)])) for k in range(P)]
    ph = int(np.argmax(m)); b = x[ok & (idx % P == ph)]; o = x[ok & (idx % P != ph)]
    return {"phase": ph, "ratio": round(float(b.mean() / o.mean()), 2), "boundary": round(float(b.mean()), 2), "other": round(float(o.mean()), 2)}


ap = argparse.ArgumentParser()
for k in ("gen", "gen-lm", "src", "src-lm"): ap.add_argument("--" + k, required=True)
ap.add_argument("--period", type=int, default=13); ap.add_argument("--out", default=""); a = ap.parse_args()
gE, gB = series(a.gen, a.gen_lm); sE, sB = series(a.src, a.src_lm); r = {}
for P in (a.period, 12):
    r[f"P{P}"] = {"gen_eye": fold(gE, P), "gen_band": fold(gB, P), "src_eye": fold(sE, P), "src_band": fold(sB, P)}
if a.out: json.dump(r, open(a.out, "w"))
print(json.dumps(r))
