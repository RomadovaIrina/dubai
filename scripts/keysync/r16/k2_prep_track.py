#!/usr/bin/env python3
"""r16 K2: TRACKED face crop for KeySync (Q's q05 / q06: ks_prep's static box = 1.7 face_h + the whole head movement -> the face is 0.30-0.36 of
the 512 input and KeySync's lower face turns to garbage; fine at >= 0.39). Instead: a square box of CONSTANT side = --k x the face height
(p90; default 1.8 -> the face ~0.55 of the input) + 2 x the smoothing residual, whose centre follows the face centre smoothed over time
(gaussian, sigma --sigma frames), clamped inside the frame. One KeySync job, no seams; the paste uses the box of each master frame
(ks_plain.py reads "xs" / "ys" lists in box.json). Writes <out>.mp4 (512, 25 fps, the KeySync input) and <out>_box.json.
    python k2_prep_track.py --master master_25fps.mp4 --lm source_lm_F.npz --out crop [--k 1.8 --sigma 6]"""
import argparse, json, subprocess
import numpy as np, cv2

ap = argparse.ArgumentParser(); ap.add_argument("--master", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--k", type=float, default=1.8); ap.add_argument("--sigma", type=float, default=6.0); a = ap.parse_args()
z = np.load(a.lm); LM = z["lm"][..., :2].astype(np.float64); J = z["j"] if "j" in z.files else np.arange(len(LM))
pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", a.master], capture_output=True, text=True).stdout.strip().split(",")
W, H = int(pr[0]), int(pr[1]); n = len(J)
C = np.full((n, 2), np.nan); FH = np.full(n, np.nan)
for k in range(n):
    p = LM[min(int(J[k]), len(LM) - 1)]
    if np.isfinite(p).all() and p.max() > 0:
        mn, mx = p.min(0), p.max(0); C[k] = (mn + mx) / 2; FH[k] = mx[1] - mn[1]
ok = np.isfinite(C[:, 0])
for d in (0, 1): C[:, d] = np.interp(np.arange(n), np.where(ok)[0], C[ok, d])   # fill the frames without a face
fh = float(np.nanpercentile(FH, 90))
r = int(3 * a.sigma); g = np.exp(-0.5 * (np.arange(-r, r + 1) / a.sigma) ** 2); g /= g.sum()
S = np.stack([np.convolve(np.pad(C[:, d], r, mode="edge"), g, mode="valid") for d in (0, 1)], 1)
res = float(np.percentile(np.abs(C - S).max(1), 98))
side = int(min(W, H, a.k * fh + 2 * res)) // 2 * 2
xs = np.clip(np.round(S[:, 0] - side / 2), 0, W - side).astype(int); ys = np.clip(np.round(S[:, 1] - side / 2 + 0.05 * side), 0, H - side).astype(int)
static_side = int(min(W, H, 1.7 * fh + max(np.ptp(C[ok, 0]), np.ptp(C[ok, 1]))))
box = {"x": int(xs[0]), "y": int(ys[0]), "side": side, "W": W, "H": H, "face_h": fh, "xs": xs.tolist(), "ys": ys.tolist(), "tracked": True,
       "face_frac": round(fh / side, 3), "static_side": static_side, "static_face_frac": round(fh / max(static_side, 1), 3), "smooth_residual_px": round(res, 1)}
json.dump(box, open(a.out + "_box.json", "w"))
w = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "512x512", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "12",
                      "-pix_fmt", "yuv420p", a.out + ".mp4"], stdin=subprocess.PIPE)
c = cv2.VideoCapture(a.master); k = 0
while k < n:
    okf, f = c.read()
    if not okf: break
    w.stdin.write(cv2.resize(f[ys[k]:ys[k] + side, xs[k]:xs[k] + side], (512, 512), interpolation=cv2.INTER_LANCZOS4).tobytes()); k += 1
w.stdin.close(); w.wait()
print("K2_TRACK " + json.dumps({k_: box[k_] for k_ in ("side", "face_h", "face_frac", "static_side", "static_face_frac", "smooth_residual_px")}), flush=True)
