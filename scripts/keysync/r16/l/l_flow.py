#!/usr/bin/env python3
"""r16 L: pixel motion of the lower face OUTSIDE the lips (cheek / jaw band), from dense optical flow (Farneback, 512 crop).
Band = the lower face-oval polygon (below the nose) minus the outer-lip hull dilated by 0.25 IOD; head motion removed by subtracting the
median flow of the upper face (eyes / forehead). Per frame: v = mean |residual flow| in the band (px / frame), acc = mean |flow_t - flow_t-1|
(the 2nd-order change = "inflate / deflate" speed). Output ratios gen / src over [t0, t1).
    python l_flow.py --gen G.mp4 --gen-lm G.npz --src S.mp4 --src-lm S.npz --t0 1 --t1 5 --out F.json"""
import argparse, json
import numpy as np, cv2

LOW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 352, 346, 347, 348, 349, 329, 371, 266, 425, 411, 427, 207, 187, 205, 36, 142, 101, 118, 117, 116, 123]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
UP = [10, 109, 67, 103, 54, 21, 162, 127, 234, 33, 133, 168, 362, 263, 454, 356, 389, 251, 284, 332, 297, 338]


def run(vid, lmf, t0, t1):
    lm = np.load(lmf)["lm"][:, :, :2]; cap = cv2.VideoCapture(vid); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    a, b = int(t0 * 25), min(int(t1 * 25), len(fr), len(lm)); V, A = [], []; prev = None
    for t in range(a, b - 1):
        p = lm[t]; iod = np.linalg.norm(p[263] - p[33])
        band = np.zeros((512, 512), np.uint8); cv2.fillPoly(band, [cv2.convexHull(p[LOW].astype(np.int32))], 1)
        lips = np.zeros_like(band); cv2.fillPoly(lips, [cv2.convexHull(p[LIPS].astype(np.int32))], 1)
        k = max(3, int(0.25 * iod)) | 1; lips = cv2.dilate(lips, np.ones((k, k), np.uint8))
        band = band.astype(bool) & ~lips.astype(bool)
        up = np.zeros_like(lips); cv2.fillPoly(up, [cv2.convexHull(p[UP].astype(np.int32))], 1); up = up.astype(bool) & ~band
        fl = cv2.calcOpticalFlowFarneback(fr[t], fr[t + 1], None, 0.5, 3, 21, 3, 5, 1.1, 0)
        fl = fl - np.median(fl[up], axis=0)
        V.append(float(np.linalg.norm(fl[band], axis=1).mean()))
        if prev is not None: A.append(float(np.linalg.norm((fl - prev)[band], axis=1).mean()))
        prev = fl
    np.save(vid + ".band_acc.npy", np.array(A)); np.save(vid + ".band_v.npy", np.array(V))
    return {"v": round(float(np.mean(V)), 3), "acc": round(float(np.mean(A)), 3), "acc95": round(float(np.percentile(A, 95)), 3)}


ap = argparse.ArgumentParser()
for k in ("gen", "gen-lm", "src", "src-lm", "out"): ap.add_argument("--" + k, required=True)
ap.add_argument("--t0", type=float, default=1); ap.add_argument("--t1", type=float, default=5); a = ap.parse_args()
G, S = run(a.gen, a.gen_lm, a.t0, a.t1), run(a.src, a.src_lm, a.t0, a.t1)
r = {"gen": G, "src": S, "ratio": {k: round(G[k] / max(S[k], 1e-6), 2) for k in G}}
ga, sa = np.load(a.gen + ".band_acc.npy"), np.load(a.src + ".band_acc.npy"); n = min(len(ga), len(sa))   # same timeline: frame-paired spikes
fr = ga[:n] / np.maximum(sa[:n], 0.3); top = np.argsort(-fr)[:5]
r["spike"] = {"top5_ratio": round(float(fr[top].mean()), 2), "frac_over_1.5": round(float((fr > 1.5).mean()), 3),
              "top_t": [round(a.t0 + (int(i) + 2) / 25, 2) for i in sorted(top)]}
json.dump(r, open(a.out, "w")); print(json.dumps(r))
