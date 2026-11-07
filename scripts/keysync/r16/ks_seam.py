#!/usr/bin/env python3
"""r16 K1: KeySync SEGMENT-BOUNDARY jumps. KeySync generates 13-frame segments between keyframes; at the boundary the face (texture + shape)
can step in one frame (old03: the lower-face jump at one phase of 13 is ~8x the other phases; the eye sees a "slide-show").
1. Find the phase: the lower-face frame-to-frame change of the layer minus the source crop's, averaged per phase mod --period.
   Apply only if the peak phase is > --ratio x the median of the other phases (else the video has no seam problem: unchanged).
2. At every boundary (layer frame A = K-1, B = K): replace A and B by the motion-compensated interpolations I(1/3) and I(2/3) between them
   (DIS flow both ways, each side warped by its fraction, blended), so the step is spread over 2 frames.
    ks_seam.py --layer L.mp4 --src SRC_crop.mp4 --out OUT.mp4 [--period 13 --ratio 2.0]"""
import argparse, json, subprocess, numpy as np, cv2, ks_cms
ap = argparse.ArgumentParser(); [ap.add_argument("--" + k, required=True) for k in ("layer", "src", "out")]
ap.add_argument("--period", type=int, default=13); ap.add_argument("--ratio", type=float, default=2.0); ap.add_argument("--force-phase", type=int, default=-1); a = ap.parse_args()
def rd(p, fmt="bgr24", ch=3):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", p] + (ks_cms.dec() if fmt == "bgr24" else ["-vf", "scale=512:512"]) + ["-f", "rawvideo", "-pix_fmt", fmt, "-"], capture_output=True, check=True).stdout
    return np.frombuffer(r, np.uint8).reshape(-1, 512, 512, ch) if ch > 1 else np.frombuffer(r, np.uint8).reshape(-1, 512, 512)
F = rd(a.layer); S = rd(a.src, "gray", 1); n = len(F); m = min(n, len(S))
G = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in F]
lo = (slice(280, 460), slice(140, 372))
jk = np.array([np.abs(G[t + 1][lo].astype(np.float32) - G[t][lo]).mean() for t in range(m - 1)])
js = np.array([np.abs(S[t + 1][lo].astype(np.float32) - S[t][lo]).mean() for t in range(m - 1)]); ex = jk - js
ph = np.array([ex[np.arange(len(ex)) % a.period == q].mean() for q in range(a.period)]); q = int(np.argmax(ph)) if a.force_phase < 0 else a.force_phase
rest = np.median(np.delete(ph, q)); r = float(ph[q] / max(rest, 0.05))
info = {"phase": q, "peak_excess": round(float(ph[q]), 3), "rest_median": round(float(rest), 3), "ratio": round(r, 2)}
out = F.copy(); nb = 0
if r >= a.ratio or a.force_phase >= 0:
    DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); GY, GX = np.mgrid[0:512, 0:512].astype(np.float32)
    def interp(A, B, gA, gB, tau):   # backward-warp approximation of the in-between at tau (0 = A, 1 = B)
        fab = DIS.calc(gA, gB, None); fba = DIS.calc(gB, gA, None)
        wa = cv2.remap(A, GX - tau * fab[..., 0], GY - tau * fab[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        wb = cv2.remap(B, GX - (1 - tau) * fba[..., 0], GY - (1 - tau) * fba[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        return np.clip((1 - tau) * wa + tau * wb + 0.5, 0, 255).astype(np.uint8)
    for t in range(q, n - 1, a.period):   # jump t -> t+1
        A, B = F[t], F[t + 1]; out[t] = interp(A, B, G[t], G[t + 1], 1 / 3); out[t + 1] = interp(A, B, G[t], G[t + 1], 2 / 3); nb += 1
info["boundaries_fixed"] = nb
p_ = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "512x512", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "10"] + ks_cms.enc() + [a.out], stdin=subprocess.PIPE)
p_.stdin.write(out.tobytes()); p_.stdin.close(); p_.wait(); json.dump(info, open(a.out + ".seam.json", "w")); print("== ks_seam", json.dumps(info), "->", a.out)
