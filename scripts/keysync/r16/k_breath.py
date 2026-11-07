#!/usr/bin/env python3
"""r16 K1: image-based size "breathing" of a pasted final vs the source master. Per frame: the paste box of both (gray, 256 px), DIS optical flow
final -> source, then a robust similarity fit (trimmed least squares) of the flow inside a region (source landmarks):
'lower' = face oval below the eye line minus a 0.6 IOD mouth zone (cheeks + jaw, where KeySync's pixels are); 'ring' = +-0.12 IOD band on
the lower face outline (chin / jaw against neck / background). Scale s(t); breathing = std of s minus its 1 s moving mean (%).
    python k_breath.py --final F.mp4 --master M.mp4 --box box.json --lm source_lm_F.npz --out x.npz [--label L]"""
import argparse, json, subprocess, numpy as np, cv2
ap = argparse.ArgumentParser(); [ap.add_argument("--" + k, required=True) for k in ("final", "master", "box", "lm", "out")]; ap.add_argument("--label", default=""); a = ap.parse_args()
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
b = json.load(open(a.box)); x0, y0, s = b["x"], b["y"], b["side"]; h = 256; c = h / s
z = np.load(a.lm); LM = z["lm"][..., :2].astype(np.float32); J = z["j"]
DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); GY, GX = np.mgrid[0:h, 0:h].astype(np.float32)
def frames(p):
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-i", p, "-vf", f"crop={s}:{s}:{x0}:{y0},scale={h}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    while True:
        r = pr.stdout.read(h * h)
        if len(r) < h * h: break
        yield np.frombuffer(r, np.uint8).reshape(h, h)
def simfit(P, Q):   # Q ~ s R P + t, trimmed twice
    keep = np.ones(len(P), bool)
    for _ in range(3):
        p, q = P[keep], Q[keep]; pm, qm = p.mean(0), q.mean(0); pc, qc = p - pm, q - qm
        A = (pc * qc).sum(); B = (pc[:, 0] * qc[:, 1] - pc[:, 1] * qc[:, 0]).sum(); den = (pc ** 2).sum()
        sc, th = np.hypot(A, B) / den, np.arctan2(B, A); R = sc * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        res = np.linalg.norm(Q - (P - pm) @ R.T - qm, axis=1); keep = res < np.percentile(res, 80)
    return sc
out = {"lower": [], "ring": []}
for k, (F, M) in enumerate(zip(frames(a.final), frames(a.master))):
    p = ((LM[min(int(J[min(k, len(J) - 1)]), len(LM) - 1)] - [x0, y0]) * c).astype(np.float32)
    if not np.isfinite(p).all():
        for r_ in out.values(): r_.append(np.nan)
        continue
    iod = float(np.linalg.norm(p[33] - p[263])); ov = np.zeros((h, h), np.uint8); cv2.fillPoly(ov, [cv2.convexHull(p[OVAL]).astype(np.int32)[:, 0]], 1)
    mz = np.zeros_like(ov); cv2.fillPoly(mz, [cv2.convexHull(p[LIPS]).astype(np.int32)[:, 0]], 1); mz = cv2.dilate(mz, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.6 * iod) | 1,) * 2))
    lo = ov.copy(); lo[: int(p[[33, 263]][:, 1].mean() + 0.2 * iod)] = 0; lo[mz > 0] = 0
    r = max(2, int(0.12 * iod)); ring = cv2.dilate(ov, np.ones((r, r), np.uint8)) - cv2.erode(ov, np.ones((r, r), np.uint8)); ring[: int(p[1][1])] = 0
    fl = DIS.calc(F, M, None)   # final -> source
    for nm, mk in (("lower", lo), ("ring", ring)):
        ys, xs = np.nonzero(mk)
        if len(xs) < 50: out[nm].append(np.nan); continue
        P = np.stack([xs, ys], 1).astype(np.float64); Q = P + fl[ys, xs].astype(np.float64); out[nm].append(float(simfit(P, Q)))
res = {"label": a.label}
for nm, v in out.items():
    v = np.array(v); ok = np.isfinite(v); x = np.interp(np.arange(len(v)), np.flatnonzero(ok), v[ok])
    bb = x - np.convolve(np.pad(x, (12, 12), mode="edge"), np.ones(25) / 25, "valid"); out[nm] = x
    res[nm] = {"breath_pct": round(float(bb.std() * 100), 3), "p95_pct": round(float(np.percentile(np.abs(bb), 95) * 100), 3), "mean": round(float(x.mean()), 4)}
np.savez(a.out, **out); print(json.dumps(res))
