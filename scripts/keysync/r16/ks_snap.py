#!/usr/bin/env python3
"""r16 K1: KeySync mouth SNAPS (the mouth opens / closes fully within one frame, e.g. m18 18.48 -> 18.52 s). On the 25 fps KeySync layer (512):
frame k is a snap if |open[k] - open[k-1]| > --thr (open = lips 13-14 / IOD from the layer's landmarks) and the change continues in the same
direction or settles (not a one-frame spike, which ks_tsmooth handles). It is replaced, inside the feathered lower-face mask, by the flow
in-between of frames k-1 and k+1 (DIS flow both ways, each warped half way, blended), so the open / close takes 2 frames instead of 1.
    ks_snap.py --layer L.mp4 --lm L.mediapipe.npz --out OUT.mp4 [--thr 0.09]"""
import argparse, json, subprocess, numpy as np, cv2, ks_cms
ap = argparse.ArgumentParser(); ap.add_argument("--layer", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--thr", type=float, default=0.09); a = ap.parse_args()
F = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", a.layer] + ks_cms.dec() + ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout, np.uint8).reshape(-1, 512, 512, 3).copy()
L = np.load(a.lm)["lm"][..., :2].astype(np.float32); n = min(len(F), len(L))
iod = np.linalg.norm(L[:, 33] - L[:, 263], axis=1); op = np.linalg.norm(L[:, 13] - L[:, 14], axis=1) / iod; d = np.diff(op[:n])
LOW = [1, 2, 98, 327, 205, 425, 50, 280, 187, 411, 192, 416, 214, 434, 138, 367, 135, 364, 169, 394, 170, 395, 140, 369, 171, 396, 175, 152, 148, 377, 176, 400, 149, 378, 150, 379, 136, 365, 172, 397, 58, 288]
DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); GY, GX = np.mgrid[0:512, 0:512].astype(np.float32)
snaps = [k for k in range(1, n - 1) if np.isfinite(d[k - 1]) and abs(d[k - 1]) > a.thr and not (np.sign(d[k]) == -np.sign(d[k - 1]) and abs(d[k]) > a.thr)]
out = F.copy()
for k in snaps:
    g0, g2 = cv2.cvtColor(F[k - 1], cv2.COLOR_BGR2GRAY), cv2.cvtColor(F[k + 1], cv2.COLOR_BGR2GRAY)
    f02 = DIS.calc(g0, g2, None); f20 = DIS.calc(g2, g0, None)
    w0 = cv2.remap(F[k - 1], GX - 0.5 * f02[..., 0], GY - 0.5 * f02[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)   # approx: backward sampling with half flow
    w2 = cv2.remap(F[k + 1], GX - 0.5 * f20[..., 0], GY - 0.5 * f20[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mid = 0.5 * w0.astype(np.float32) + 0.5 * w2.astype(np.float32)
    m = np.zeros((512, 512), np.float32); cv2.fillConvexPoly(m, cv2.convexHull(np.round(L[k, LOW]).astype(np.int32)), 1.0)
    m = cv2.GaussianBlur(cv2.dilate(m, np.ones((21, 21), np.uint8)), (0, 0), 8)[..., None]
    out[k] = (F[k] * (1 - m) + mid * m + 0.5).astype(np.uint8)
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "512x512", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "10"] + ks_cms.enc() + [a.out], stdin=subprocess.PIPE)
p.stdin.write(out.tobytes()); p.stdin.close(); p.wait()
json.dump({"snaps": [int(k) for k in snaps], "t_s": [round(k / 25, 2) for k in snaps]}, open(a.out + ".snaps.json", "w"))
print(f"== ks_snap {n} frames, thr {a.thr}: {len(snaps)} snap frames {[round(k / 25, 2) for k in snaps]} -> {a.out}")
