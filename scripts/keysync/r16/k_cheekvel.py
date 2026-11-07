#!/usr/bin/env python3
"""r16 K1: how fast the cheeks / jaw contour move (non-rigid) in a 512 crop video vs the source crop. Per frame t: DIS flow t -> t+1 (256 px),
minus the rigid head motion (a similarity fit on the upper face: eyes / brows / nose bridge band); the residual flow magnitude, mean over the
cheek + jaw band (face oval below the eye line, minus the lips dilated 0.35 IOD) = the cheek speed (px/frame at 512).
Prints mean / p95 for the video and for the source, the share of frames above the source's p95, and the worst seconds.
    python k_cheekvel.py --ks KS.mp4 --ks-lm KS.mediapipe.npz --src SRC_crop.mp4 --src-lm SRC_crop.mediapipe.npz [--out x.npz]"""
import argparse, json, subprocess, numpy as np, cv2
ap = argparse.ArgumentParser(); [ap.add_argument("--" + k, required=True) for k in ("ks", "ks-lm", "src", "src-lm")]; ap.add_argument("--out", default=""); ap.add_argument("--n", type=int, default=0)
a = ap.parse_args()
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
UP = [33, 263, 133, 362, 168, 6, 70, 300, 105, 334, 197]
h = 256; c = h / 512.0; DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); GY, GX = np.mgrid[0:h, 0:h].astype(np.float32)
def rd(p):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-vf", f"scale={h}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
    return np.frombuffer(r, np.uint8).reshape(-1, h, h)
def speed(V, L, n):
    out = np.full(n, np.nan)
    for t in range(n - 1):
        p = L[t] * c
        if not np.isfinite(p).all(): continue
        iod = float(np.linalg.norm(p[33] - p[263])); fl = DIS.calc(V[t], V[t + 1], None)
        P = p[UP].astype(np.float64); Q = P + fl[np.clip(P[:, 1].astype(int), 0, h - 1), np.clip(P[:, 0].astype(int), 0, h - 1)]
        M, _ = cv2.estimateAffinePartial2D(P.astype(np.float32), Q.astype(np.float32))
        if M is None: continue
        rig = np.stack([M[0, 0] * GX + M[0, 1] * GY + M[0, 2] - GX, M[1, 0] * GX + M[1, 1] * GY + M[1, 2] - GY], -1)
        m = np.zeros((h, h), np.uint8); cv2.fillPoly(m, [cv2.convexHull(p[OVAL].astype(np.float32)).astype(np.int32)[:, 0]], 1); m[: int(p[[33, 263], 1].mean() + 0.3 * iod)] = 0
        lz = np.zeros_like(m); cv2.fillPoly(lz, [cv2.convexHull(p[LIPS].astype(np.float32)).astype(np.int32)[:, 0]], 1); lz = cv2.dilate(lz, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.7 * iod) | 1,) * 2))
        m[lz > 0] = 0
        if m.sum() < 50: continue
        out[t] = float(np.linalg.norm(fl - rig, axis=-1)[m > 0].mean()) / c
    return out
K, S = rd(a.ks), rd(a.src); LK = np.load(a.ks_lm)["lm"][..., :2].astype(np.float32); LS = np.load(a.src_lm)["lm"][..., :2].astype(np.float32)
n = a.n or min(len(K), len(S), len(LK), len(LS)); vk, vs = speed(K, LK, n), speed(S, LS, n)
p95 = np.nanpercentile(vs, 95); hi = vk > p95
sec = [(round(float(np.nanmean(vk[i:i + 25])), 2), i / 25) for i in range(0, n - 25, 25)]
res = {"ks_mean": round(float(np.nanmean(vk)), 3), "ks_p95": round(float(np.nanpercentile(vk, 95)), 3), "src_mean": round(float(np.nanmean(vs)), 3), "src_p95": round(float(p95), 3),
       "share_above_src_p95": round(float(np.nanmean(hi)), 3), "worst_s": [s for _, s in sorted(sec, reverse=True)[:6]]}
if a.out: np.savez(a.out, ks=vk, src=vs)
print(json.dumps(res))
