#!/usr/bin/env python3
"""r16c K2: texture boil ("springing") of a lip-synced render vs its source. Per frame, the cheek / beard region (face oval minus the lips + 0.25
mouth width, minus the eyes band) of BOTH videos is warped with the SOURCE landmarks to a canonical 128 x 128 frame (similarity from the eye
corners + chin), high-passed (gray - Gaussian blur sigma 2); boil_k = mean |HP_k - HP_{k-1}| inside the region. Ratio = median over frames of
boil(ours) / boil(source): ~1 = the texture is as stable as the source, > 1 = it re-rolls (boils). Also per-second ratios, to find bad stretches.
    python k2_boil.py --src SRC_25fps.mp4 --lm SRC_mp.npz --ours A.mp4 [B.mp4 ...] [--t0 T0 --t1 T1]"""
import argparse, json
import numpy as np, cv2

OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--ours", nargs="+", required=True)
ap.add_argument("--t0", type=float, default=0); ap.add_argument("--t1", type=float, default=1e9); a = ap.parse_args()
L = np.load(a.lm)["lm"][..., :2].astype(np.float32); K0, K1 = int(a.t0 * 25), min(int(a.t1 * 25), len(L))
DST = np.float32([[40, 40], [88, 40], [64, 120]])   # left eye corner, right eye corner, chin


def series(path):
    c = cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_FRAMES, K0); prev = None; out = np.full(K1 - K0, np.nan)
    for k in range(K0, K1):
        ok, f = c.read()
        if not ok: break
        p = L[k]
        if not np.isfinite(p).all(): prev = None; continue
        M = cv2.getAffineTransform(np.float32([p[33], p[263], p[152]]), DST)
        g = cv2.warpAffine(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), M, (128, 128)).astype(np.float32); hp = g - cv2.GaussianBlur(g, (0, 0), 2)
        q = cv2.transform(p[None], M)[0]; mw = np.linalg.norm(q[61] - q[291])
        m = cv2.fillPoly(np.zeros((128, 128), np.uint8), [cv2.convexHull(q[OVAL])[:, 0].astype(np.int32)], 1)
        lips = cv2.dilate(cv2.fillPoly(np.zeros((128, 128), np.uint8), [cv2.convexHull(q[LIPS])[:, 0].astype(np.int32)], 1), np.ones((int(0.5 * mw) | 1,) * 2, np.uint8))
        m[lips > 0] = 0; m[: int(q[1, 1]) - 6] = 0   # below the nose tip only: cheeks, jaw, beard
        if prev is not None and m.sum() > 50: out[k - K0] = float(np.abs(hp - prev)[m > 0].mean())
        prev = hp
    return out


S = series(a.src); res = {}
for o in a.ours:
    O = series(o); r = O / np.maximum(S, 1e-3); ok = np.isfinite(r)
    sec = [float(np.nanmedian(r[i:i + 25])) if np.isfinite(r[i:i + 25]).any() else None for i in range(0, len(r), 25)]
    res[o] = {"boil_ratio_median": round(float(np.nanmedian(r[ok])), 3), "boil_ratio_p90": round(float(np.nanpercentile(r[ok], 90)), 3),
              "ours_abs": round(float(np.nanmedian(O)), 3), "src_abs": round(float(np.nanmedian(S)), 3), "per_s": [round(x, 2) if x else None for x in sec]}
    print(f"K2_BOIL {o.split('/')[-1]:32s} ratio median {res[o]['boil_ratio_median']:.3f}  p90 {res[o]['boil_ratio_p90']:.3f}  (ours {res[o]['ours_abs']:.2f} / src {res[o]['src_abs']:.2f})", flush=True)
json.dump(res, open("boil.json", "w"))
