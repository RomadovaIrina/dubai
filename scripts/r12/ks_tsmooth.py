#!/usr/bin/env python3
# r13 LS: copied unchanged from origin/experiment/r13-keysync e4bc43f scripts/keysync/ks_tsmooth.py (run on the LatentSync layer)
"""r12 KS 'лаги' fix: temporal smoothing of the mouth GEOMETRY on the 25 fps KeySync layer.
For frame t: Farneback flow t->t-1 (F-) and t->t+1 (F+) inside the lower face; d = (F- + F+) / 2 is where the neighbours' mean puts each pixel
(the second difference = the jerk); frame t is warped by alpha * d, so a one-frame jerk is pulled to the neighbours' mean while smooth motion
(F- ~ -F+, d ~ 0) is untouched. Applied only inside the lower-face mask (layer landmarks below the nose, dilated, feathered).
--alpha auto (adaptive): alpha_t = amax * clip((rK - rS) / ramp, 0, 1), where rK / rS = the mouth jerk-spike rate of the layer / of the source
(--src-lm: landmarks of the 25 fps master, same frames) in a --win frame window centred on t (spike = ks_lag.py's: jerk > 4 x the source median),
gaussian-smoothed; so only the stretches where KeySync jitters more than the source get smoothed, the rest stays untouched.
    ks_tsmooth.py --layer L.mp4 --lm L.mediapipe.npz --out OUT.mp4 [--alpha 0.7 | --alpha auto --src-lm MASTER.mediapipe.npz] [--passes 1]"""
import argparse, json, subprocess, numpy as np, cv2, ks_cms
ap = argparse.ArgumentParser(); ap.add_argument("--layer"); ap.add_argument("--lm"); ap.add_argument("--out"); ap.add_argument("--alpha", default="0.7")
ap.add_argument("--passes", type=int, default=1); ap.add_argument("--src-lm"); ap.add_argument("--amax", type=float, default=0.8)
ap.add_argument("--win", type=int, default=25); ap.add_argument("--ramp", type=float, default=0.1); a = ap.parse_args()
pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", a.layer], capture_output=True, text=True).stdout.strip().split(",")
W, H = int(pr[0]), int(pr[1])
F = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", a.layer] + ks_cms.dec() + ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout, np.uint8).reshape(-1, H, W, 3).copy()
d = np.load(a.lm); lm = d["lm"][..., :2].astype(np.float32)
if lm.max() < 2: lm = lm * np.array([W, H], np.float32)
n = min(len(F), len(lm))
def jerk(l):   # ks_lag.py's per-frame mouth jerks: opening and lip centre relative to the nose, / IOD
    l = l.astype(np.float64); iod = np.linalg.norm(l[:, 33] - l[:, 263], axis=1)
    op = np.linalg.norm(l[:, 13] - l[:, 14], axis=1) / iod; c = l[:, [61, 291, 13, 14]].mean(1) / iod[:, None] - l[:, 1] / iod[:, None]
    return np.pad(np.abs(np.diff(op, 2)), (1, 1)), np.pad(np.linalg.norm(np.diff(c, 2, axis=0), axis=1), (1, 1))
if a.alpha == "auto":
    sl = np.load(a.src_lm)["lm"][..., :2]; m = min(n, len(sl))
    jo, jc = jerk(lm[:m]); so, sc = jerk(sl[:m]); to, tc = 4 * np.median(so[so > 0]), 4 * np.median(sc[sc > 0])
    k = np.ones(a.win) / a.win; rK = np.convolve(((jo > to) | (jc > tc)).astype(float), k, "same"); rS = np.convolve(((so > to) | (sc > tc)).astype(float), k, "same")
    al = a.amax * np.clip((rK - rS) / a.ramp, 0, 1); al = cv2.GaussianBlur(al.reshape(1, -1).astype(np.float32), (0, 0), 3).ravel()
    AL = np.zeros(n, np.float32); AL[:m] = al
    json.dump({"mean_alpha": round(float(AL.mean()), 3), "frac_on": round(float((AL > 0.05).mean()), 3), "rate_layer": round(float(rK.mean()), 4), "rate_src": round(float(rS.mean()), 4)},
              open(a.out + ".alpha.json", "w")); np.save(a.out + ".alpha.npy", AL)
else:
    AL = np.full(n, float(a.alpha), np.float32)
LOW = [1, 2, 98, 327, 205, 425, 50, 280, 187, 411, 192, 416, 214, 434, 138, 367, 135, 364, 169, 394, 170, 395, 140, 369, 171, 396, 175, 152, 148, 377, 176, 400, 149, 378, 150, 379, 136, 365, 172, 397, 58, 288]
def mask(i):
    p = lm[i, LOW]; x0, y0 = np.floor(p.min(0)).astype(int) - 24; x1, y1 = np.ceil(p.max(0)).astype(int) + 24
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W), min(y1, H)
    m = np.zeros((H, W), np.float32); cv2.fillConvexPoly(m, cv2.convexHull(np.round(p).astype(np.int32)), 1.0)
    m = cv2.GaussianBlur(cv2.dilate(m, np.ones((15, 15), np.uint8)), (0, 0), 6); return m, (x0, y0, x1, y1)
for _ in range(a.passes):
    G = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in F[:n]]; out = F.copy()
    for t in range(1, n - 1):
        if AL[t] < 0.02: continue
        m, (x0, y0, x1, y1) = mask(t)
        g0, gm, gp = G[t][y0:y1, x0:x1], G[t - 1][y0:y1, x0:x1], G[t + 1][y0:y1, x0:x1]
        fm = cv2.calcOpticalFlowFarneback(g0, gm, None, 0.5, 3, 15, 3, 5, 1.2, 0); fp = cv2.calcOpticalFlowFarneback(g0, gp, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dd = AL[t] * 0.5 * (fm + fp); yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
        w = cv2.remap(F[t][y0:y1, x0:x1], xx - dd[..., 0], yy - dd[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        mm = m[y0:y1, x0:x1, None]; out[t][y0:y1, x0:x1] = (F[t][y0:y1, x0:x1] * (1 - mm) + w * mm + 0.5).astype(np.uint8)
    F = out
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "10"] + ks_cms.enc() + [a.out], stdin=subprocess.PIPE)
p.stdin.write(F.tobytes()); p.stdin.close(); p.wait(); print(f"== ks_tsmooth {n} frames alpha {a.alpha} (mean {AL.mean():.3f}) passes {a.passes} -> {a.out}")
