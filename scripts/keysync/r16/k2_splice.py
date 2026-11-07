#!/usr/bin/env python3
"""r16 K2 variant (b): more KeySync interp steps only on WIDE-OPEN frames. Base = the 2-step render; frames whose generated opening
(gap / width, the base's landmarks) > --open get the N-step render, with a 2-frame linear ramp each side (no pop). Prints the share of
frames taken from the N-step render and the face-GPU cost if only those interpolation chunks ran at N steps.
    python k2_splice.py --base s2.mp4 --base-lm s2.npz --hi s8.mp4 --out b.mp4 [--open 0.18] [--chunk 14] [--cost-base 137 --cost-hi 230]"""
import argparse, subprocess, json
import numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--base-lm", required=True); ap.add_argument("--hi", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--open", type=float, default=0.18); ap.add_argument("--chunk", type=int, default=14)
ap.add_argument("--cost-base", type=float, default=0); ap.add_argument("--cost-hi", type=float, default=0); a = ap.parse_args()
L = np.load(a.base_lm)["lm"][..., :2]
op = np.array([np.linalg.norm(p[13] - p[14]) / max(np.linalg.norm(p[61] - p[291]), 1) if np.isfinite(p).all() else 0 for p in L])
hit = op > a.open; w = hit.astype(np.float32)
for d in (1, 2):
    w = np.maximum(w, np.maximum(np.roll(hit, d), np.roll(hit, -d)) * (1 - d / 3.0))
cb, ch = cv2.VideoCapture(a.base), cv2.VideoCapture(a.hi); fps = cb.get(cv2.CAP_PROP_FPS)
wr = None; k = 0
while True:
    ob, fb = cb.read(); oh, fh = ch.read()
    if not ob: break
    if wr is None: wr = cv2.VideoWriter(a.out + ".tmp.avi", cv2.VideoWriter_fourcc(*"FFV1"), fps, (fb.shape[1], fb.shape[0]))
    x = w[k] if k < len(w) else 0
    wr.write(fb if (x == 0 or not oh) else np.clip(fb * (1 - x) + fh * x + 0.5, 0, 255).astype(np.uint8)); k += 1
wr.release(); subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.out + ".tmp.avi", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", a.out], check=True)
subprocess.run(["rm", "-f", a.out + ".tmp.avi"])
ch_ = np.array([w[i:i + a.chunk].max() > 0 for i in range(0, len(w), a.chunk)])
r = {"frames_hi": round(float((w > 0).mean()), 3), "chunks_hi": round(float(ch_.mean()), 3)}
if a.cost_base and a.cost_hi: r["est_face_gpu_s_vmin"] = round(a.cost_base + ch_.mean() * (a.cost_hi - a.cost_base), 1)
print("K2_SPLICE " + json.dumps(r))
