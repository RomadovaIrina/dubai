#!/usr/bin/env python3
"""r16 K1: image-based per-frame registration of the raw KeySync 512 crop onto the source 512 crop (same frame index), ECC affine at 256 px,
inside the face oval minus the mouth / jaw zone (KeySync's generated motion stays out of the fit). Output npz: M (n, 2, 3) mapping KS -> source
in 512 coords (nan where it failed), plus scale s(t) = sqrt(|det A|), and prints the scale "breathing" (std of s minus its 1 s moving mean, %).
    python k_ecc.py --ks KS.mp4 --src SRC_crop.mp4 --lm SRC_crop.mediapipe.npz --out ecc.npz"""
import argparse, subprocess, numpy as np, cv2
ap = argparse.ArgumentParser(); [ap.add_argument("--" + k, required=True) for k in ("ks", "src", "lm", "out")]; a = ap.parse_args()
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
def rd(p):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(r, np.uint8).reshape(-1, 512, 512)
K, S = rd(a.ks), rd(a.src); L = np.load(a.lm)["lm"][..., :2].astype(np.float32); n = min(len(K), len(S), len(L))
M = np.full((n, 2, 3), np.nan, np.float32); h = 256; c = h / 512.0; crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5); prev = np.eye(2, 3, dtype=np.float32)
for i in range(n):
    p = L[i]
    if not np.isfinite(p).all(): continue
    iod = float(np.linalg.norm(p[33] - p[263])); m = np.zeros((512, 512), np.uint8)
    cv2.fillPoly(m, [cv2.convexHull(p[OVAL]).astype(np.int32)[:, 0]], 1); m = cv2.erode(m, np.ones((9, 9), np.uint8))
    mz = np.zeros_like(m); cv2.fillPoly(mz, [cv2.convexHull(p[LIPS]).astype(np.int32)[:, 0]], 1); mz = cv2.dilate(mz, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.7 * iod) | 1,) * 2))
    yl = int(p[[61, 291], 1].mean() - 0.1 * iod); m[yl:] = 0; m[mz > 0] = 0   # nothing below the mouth corners (jaw follows the generated mouth)
    t, s_ = cv2.resize(S[i], (h, h), interpolation=cv2.INTER_AREA), cv2.resize(K[i], (h, h), interpolation=cv2.INTER_AREA)
    mk = cv2.resize(m, (h, h), interpolation=cv2.INTER_NEAREST)
    W = prev.copy()
    try:
        _, W = cv2.findTransformECC(cv2.GaussianBlur(t, (0, 0), 1).astype(np.float32), cv2.GaussianBlur(s_, (0, 0), 1).astype(np.float32), W, cv2.MOTION_AFFINE, crit, mk, 3)
    except cv2.error: continue
    # W maps template(src) coords -> input(KS) coords; invert -> KS -> src, back to 512 px
    Wi = cv2.invertAffineTransform(W); Wi[:, 2] /= c; M[i] = Wi; prev = W
sc = np.sqrt(np.abs(np.linalg.det(M[:, :, :2].astype(np.float64)))); ok = np.isfinite(sc)
x = np.interp(np.arange(n), np.flatnonzero(ok), sc[ok]); tr = np.convolve(np.pad(x, (12, 12), mode="edge"), np.ones(25) / 25, "valid")
b = x - tr; t_ = np.linalg.norm(M[:, :, 2], axis=1)
np.savez(a.out, M=M, scale=sc)
print(f"ECC {ok.sum()}/{n} frames: scale mean {np.nanmean(sc):.4f}, breathing std {b.std() * 100:.3f} % p95 {np.percentile(np.abs(b), 95) * 100:.3f} %, |t| mean {np.nanmean(t_):.2f} px")
