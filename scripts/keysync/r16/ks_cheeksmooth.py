#!/usr/bin/env python3
"""r16 K1: slow down KeySync's cheek / jaw contour ("лицо раздувается и сжимается, мимика слишком быстрая") WITHOUT touching the lips.
On the 25 fps KeySync layer (512): for frame t, DIS flows t -> t+-1..t+-R; d = the gaussian-weighted mean of these flows = where the neighbours
put each pixel (symmetric, so steady motion gives d ~ 0); frame t is warped by alpha * d inside the cheek / jaw band = the lower face (below
the eyes, down to the chin + room) MINUS the lips dilated --lipdil IOD, feathered. Geometry only; the mouth keeps KeySync's exact motion (sync).
    ks_cheeksmooth.py --layer L.mp4 --lm L.mediapipe.npz --out OUT.mp4 [--alpha 0.8 --radius 2 --sigma 1.2 --lipdil 0.3]"""
import argparse, subprocess, numpy as np, cv2, ks_cms
ap = argparse.ArgumentParser(); ap.add_argument("--layer", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--alpha", type=float, default=0.8); ap.add_argument("--radius", type=int, default=2); ap.add_argument("--sigma", type=float, default=1.2)
ap.add_argument("--lipdil", type=float, default=0.3); a = ap.parse_args()
F = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", a.layer] + ks_cms.dec() + ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout, np.uint8).reshape(-1, 512, 512, 3)
L = np.load(a.lm)["lm"][..., :2].astype(np.float32); n = min(len(F), len(L))
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); GY, GX = np.mgrid[0:512, 0:512].astype(np.float32)
G = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in F[:n]]
w = {j: float(np.exp(-0.5 * (j / a.sigma) ** 2)) for j in range(-a.radius, a.radius + 1) if j}
out = F[:n].copy(); last = None
for t in range(n):
    p = L[t]
    if np.isfinite(p).all():
        iod = float(np.linalg.norm(p[33] - p[263])); m = np.zeros((512, 512), np.float32)
        ov = p[OVAL].copy(); ov[ov[:, 1] > p[152, 1] - 1, 1] += 0.12 * iod   # room below the chin (the generated jaw opens)
        cv2.fillPoly(m, [cv2.convexHull(ov).astype(np.int32)[:, 0]], 1.0); m[: int(p[[33, 263], 1].mean() + 0.35 * iod)] = 0
        lz = np.zeros((512, 512), np.uint8); cv2.fillPoly(lz, [cv2.convexHull(p[LIPS]).astype(np.int32)[:, 0]], 1); lz = cv2.dilate(lz, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(a.lipdil * iod) + 1,) * 2))
        m[lz > 0] = 0; m = cv2.GaussianBlur(cv2.dilate(m, np.ones((15, 15), np.uint8)), (0, 0), 0.06 * iod); last = m
    m = last
    if m is None: continue
    ys, xs = np.nonzero(m > 0.01); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    d = np.zeros((y1 - y0, x1 - x0, 2), np.float32); ws = 0.0
    for j, wj in w.items():
        if 0 <= t + j < n:
            d += wj * DIS.calc(G[t][y0:y1, x0:x1], G[t + j][y0:y1, x0:x1], None); ws += wj
    if ws == 0: continue
    d *= a.alpha / ws; yy, xx = GY[y0:y1, x0:x1], GX[y0:y1, x0:x1]
    wp = cv2.remap(F[t], xx + d[..., 0], yy + d[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mm = m[y0:y1, x0:x1, None]; out[t][y0:y1, x0:x1] = (F[t][y0:y1, x0:x1] * (1 - mm) + wp * mm + 0.5).astype(np.uint8)
p_ = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "512x512", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "10"] + ks_cms.enc() + [a.out], stdin=subprocess.PIPE)
p_.stdin.write(out.tobytes()); p_.stdin.close(); p_.wait(); print(f"== ks_cheeksmooth {n} frames alpha {a.alpha} radius {a.radius} -> {a.out}")
