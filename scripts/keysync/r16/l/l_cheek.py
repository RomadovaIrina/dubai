#!/usr/bin/env python3
"""r16 L: lower-face motion vs the source (512 crops, MediaPipe npz of each, mp_landmarks.py). Head motion removed per frame by a
similarity fit on stable points (eye corners, nose bridge, forehead) to the frame-0 pose; units = IOD (x1000).
  cheek = the side contour (234 93 132 58 / 454 323 361 288); jaw = the lower contour incl. chin; width = 234-454 distance (inflate/deflate).
  v = mean |p_t - p_t-1|, a95 = p95 of |p_t+1 - 2 p_t + p_t-1|, wd = std of the frame-to-frame width change.
    python l_cheek.py --gen G.npz --src S.npz --t0 1 --t1 5 [--extra label=X.npz ...] --out M.json"""
import argparse, json
import numpy as np

STAB = [33, 133, 263, 362, 168, 6, 10, 151]
CHEEK = [234, 93, 132, 58, 454, 323, 361, 288]
JAW = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397]


def norm(lm):
    P = lm[:, :, :2].astype(np.float64); ref = P[0, STAB]; out = np.empty_like(P)
    for t in range(len(P)):
        s = P[t, STAB]; ms, mr = s.mean(0), ref.mean(0); a, b = s - ms, ref - mr
        u, sv, vt = np.linalg.svd(a.T @ b); R = u @ vt; sc = sv.sum() / (a ** 2).sum()
        out[t] = (P[t] - ms) @ R * sc + mr
    iod = np.linalg.norm(ref[2] - ref[0])
    return out / iod * 1000


def stats(lm, t0, t1):
    Q = norm(lm)[int(t0 * 25):int(t1 * 25)]
    r = {}
    for nm, ix in (("cheek", CHEEK),):   # no "jaw": it moves with articulation (audit #4)
        X = Q[:, ix]; v = np.linalg.norm(X[1:] - X[:-1], axis=-1).mean(1); acc = np.linalg.norm(X[2:] - 2 * X[1:-1] + X[:-2], axis=-1).mean(1)
        r[f"{nm}_v"] = round(float(v.mean()), 2); r[f"{nm}_a95"] = round(float(np.percentile(acc, 95)), 2)
    w = np.linalg.norm(Q[:, 234] - Q[:, 454], axis=-1); r["width_dstd"] = round(float(np.diff(w).std()), 2)
    return r


ap = argparse.ArgumentParser(); ap.add_argument("--gen", required=True); ap.add_argument("--src", required=True)
ap.add_argument("--t0", type=float, default=1); ap.add_argument("--t1", type=float, default=5); ap.add_argument("--out", required=True)
a = ap.parse_args()
S = stats(np.load(a.src)["lm"], a.t0, a.t1); Gn = stats(np.load(a.gen)["lm"], a.t0, a.t1)
res = {"gen": Gn, "src": S, "ratio": {k: round(Gn[k] / max(S[k], 1e-6), 2) for k in Gn}}
json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res))
