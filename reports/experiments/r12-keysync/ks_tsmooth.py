#!/usr/bin/env python3
"""r12 KS 'лаги' fix: temporal smoothing of the mouth GEOMETRY on the 25 fps KeySync layer.
For frame t: Farneback flow t->t-1 (F-) and t->t+1 (F+) inside the lower face; d = (F- + F+) / 2 is where the neighbours' mean puts each pixel
(the second difference = the jerk); frame t is warped by alpha * d, so a one-frame jerk is pulled to the neighbours' mean while smooth motion
(F- ~ -F+, d ~ 0) is untouched. Applied only inside the lower-face mask (layer landmarks below the nose, dilated, feathered).
    ks_tsmooth.py --layer L.mp4 --lm L.mediapipe.npz --out OUT.mp4 [--alpha 0.7 --passes 1]"""
import argparse, subprocess, numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("--layer"); ap.add_argument("--lm"); ap.add_argument("--out"); ap.add_argument("--alpha", type=float, default=0.7)
ap.add_argument("--passes", type=int, default=1); a = ap.parse_args()
pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", a.layer], capture_output=True, text=True).stdout.strip().split(",")
W, H = int(pr[0]), int(pr[1])
F = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", a.layer, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout, np.uint8).reshape(-1, H, W, 3).copy()
d = np.load(a.lm); lm = d["lm"][..., :2].astype(np.float32)
if lm.max() < 2: lm = lm * np.array([W, H], np.float32)
n = min(len(F), len(lm))
LOW = [1, 2, 98, 327, 205, 425, 50, 280, 187, 411, 192, 416, 214, 434, 138, 367, 135, 364, 169, 394, 170, 395, 140, 369, 171, 396, 175, 152, 148, 377, 176, 400, 149, 378, 150, 379, 136, 365, 172, 397, 58, 288]
def mask(i):
    p = lm[i, LOW]; x0, y0 = np.floor(p.min(0)).astype(int) - 24; x1, y1 = np.ceil(p.max(0)).astype(int) + 24
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W), min(y1, H)
    m = np.zeros((H, W), np.float32); cv2.fillConvexPoly(m, cv2.convexHull(np.round(p).astype(np.int32)), 1.0)
    m = cv2.GaussianBlur(cv2.dilate(m, np.ones((15, 15), np.uint8)), (0, 0), 6); return m, (x0, y0, x1, y1)
for _ in range(a.passes):
    G = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in F[:n]]; out = F.copy()
    for t in range(1, n - 1):
        m, (x0, y0, x1, y1) = mask(t)
        g0, gm, gp = G[t][y0:y1, x0:x1], G[t - 1][y0:y1, x0:x1], G[t + 1][y0:y1, x0:x1]
        fm = cv2.calcOpticalFlowFarneback(g0, gm, None, 0.5, 3, 15, 3, 5, 1.2, 0); fp = cv2.calcOpticalFlowFarneback(g0, gp, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dd = a.alpha * 0.5 * (fm + fp); yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
        w = cv2.remap(F[t][y0:y1, x0:x1], xx - dd[..., 0], yy - dd[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mm = m[y0:y1, x0:x1, None]; out[t][y0:y1, x0:x1] = (F[t][y0:y1, x0:x1] * (1 - mm) + w * mm + 0.5).astype(np.uint8)
    F = out
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
p.stdin.write(F.tobytes()); p.stdin.close(); p.wait(); print(f"== ks_tsmooth {n} frames alpha {a.alpha} passes {a.passes} -> {a.out}")
