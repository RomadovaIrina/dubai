#!/usr/bin/env python3
"""r16 K1: per-frame sanity guard on the KeySync layer (512). Frame k is BROKEN if its lower-face change from frame k-1 is > --k x the local
median change (+-12 frames) AND > --k x the source crop's own change at k (a real fast move in the source is not flagged). A broken frame
is replaced by the motion-compensated in-between of its nearest good neighbours (k-1 and the next good frame); with no good frame after it
(the end), by the previous good frame. Only frames < --n (the master length) are checked. Prints every replaced frame (time) for the report.
    ks_guard.py --layer L.mp4 --src SRC_crop.mp4 --n N --out OUT.mp4 [--k 2.5]"""
import argparse, json, subprocess, numpy as np, cv2, ks_cms
ap = argparse.ArgumentParser(); [ap.add_argument("--" + x, required=True) for x in ("layer", "src", "out")]; ap.add_argument("--n", type=int, required=True)
ap.add_argument("--k", type=float, default=2.5); a = ap.parse_args()
F = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", a.layer] + ks_cms.dec() + ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout, np.uint8).reshape(-1, 512, 512, 3)
S = np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-i", a.src, "-vf", "scale=512:512", "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout, np.uint8).reshape(-1, 512, 512)
G = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in F]; nf = len(F); n = min(a.n, nf)
lo = (slice(280, 460), slice(140, 372)); jd = lambda X, t: float(np.abs(X[t][lo].astype(np.float32) - X[t - 1][lo]).mean())
jk = np.array([0.0] + [jd(G, t) for t in range(1, nf)]); js = np.array([0.0] + [jd(S, t) if t < len(S) else np.nan for t in range(1, nf)])
bad = np.zeros(nf, bool)
for t in range(1, n):
    loc = np.median(jk[max(1, t - 12): t + 13])
    if jk[t] > a.k * loc and (not np.isfinite(js[t]) or jk[t] > a.k * max(js[t], 0.5)): bad[t] = True
bad[n:] = True   # nothing after the master end counts as a good neighbour (the padding tail may degenerate)
import os as _os
_fk = _os.environ.get("KS_FAULT", "")
if _fk.startswith("guard:"):   # r16 TZ 11.2 / 11.3 fault injection: every k-th frame counts as broken (k = 1 / share), replaced like a real one
    _st = max(2, int(round(1 / float(_fk.split(":")[1])))); bad[np.arange(10, max(10, n - 1), _st)] = True; bad[n:] = True
DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); GY, GX = np.mgrid[0:512, 0:512].astype(np.float32)
def interp(A, B, gA, gB, tau):
    fab = DIS.calc(gA, gB, None); fba = DIS.calc(gB, gA, None)
    wa = cv2.remap(A, GX - tau * fab[..., 0], GY - tau * fab[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    wb = cv2.remap(B, GX - (1 - tau) * fba[..., 0], GY - (1 - tau) * fba[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    return np.clip((1 - tau) * wa + tau * wb + 0.5, 0, 255).astype(np.uint8)
out = F.copy(); fixed = []
for t in range(1, n):
    if not bad[t]: continue
    p = t - 1; q = next((u for u in range(t + 1, min(t + 4, n)) if not bad[u]), None)
    out[t] = interp(out[p], F[q], cv2.cvtColor(out[p], cv2.COLOR_BGR2GRAY), G[q], (t - p) / (q - p)) if q is not None else out[p]
    fixed.append(round(t / 25, 2))
p_ = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "512x512", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "10"] + ks_cms.enc() + [a.out], stdin=subprocess.PIPE)
p_.stdin.write(out.tobytes()); p_.stdin.close(); p_.wait(); json.dump({"fixed_s": fixed, "frames": int(n), "replaced": len(fixed)}, open(a.out + ".guard.json", "w"))
print(f"== ks_guard {n} frames: {len(fixed)} broken frames replaced at {fixed} -> {a.out}")
