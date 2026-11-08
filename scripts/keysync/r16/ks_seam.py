#!/usr/bin/env python3
"""r16 K1: KeySync SEGMENT-BOUNDARY jumps. KeySync generates 13-frame segments between keyframes; at the boundary the face (texture + shape)
can step in one frame (old03: the lower-face jump at one phase of 13 is ~8x the other phases; the eye sees a "slide-show").
The 13-frame grid restarts in every KeySync RUN: ks_chunk.py renders a long video in chunks (each = the reference frame + frames [a, b)),
so the phase is detected and applied PER CHUNK (--chunks DIR/chunks.json from ks_chunk split; without it the whole layer is one run).
Per run [a, b) of the layer:
1. the lower-face frame-to-frame change of the layer minus the source crop's, averaged per phase (t - a) mod --period;
   applied only if the peak phase is > --ratio x the median of the other phases (else that run has no seam problem: unchanged);
2. at every boundary of that run (layer frames t -> t+1): t and t+1 are replaced by the motion-compensated in-betweens I(1/3) and I(2/3)
   (DIS flow both ways, each side warped by its fraction, blended), so the step is spread over 3 transitions.
    ks_seam.py --layer L.mp4 --src SRC_crop.mp4 --out OUT.mp4 [--chunks chunks.json] [--period 13 --ratio 2.0 --min-run 78]"""
import argparse, json, os, subprocess, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ks_cms
ap = argparse.ArgumentParser(); [ap.add_argument("--" + k, required=True) for k in ("layer", "src", "out")]
ap.add_argument("--chunks", default="", help="ks_chunk.py chunks.json: the KeySync runs [a, b) on the joined layer (the 13-frame grid restarts in each)")
ap.add_argument("--period", type=int, default=13); ap.add_argument("--ratio", type=float, default=2.0, help="ratio where the smoothing STARTS (weight 0)")
ap.add_argument("--ratio-full", type=float, default=3.0, help="r16b K1 (audit P1): ratio where the weight reaches 1; linear in between, so near-identical runs give near-identical output (no 0 / 82-boundary flip)"); ap.add_argument("--force-phase", type=int, default=-1)
ap.add_argument("--min-run", type=int, default=78, help="a run shorter than this (6 periods) is left unchanged (too few boundaries to detect the phase)")
a = ap.parse_args()
def rd(p, fmt="bgr24", ch=3):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", p] + (ks_cms.dec() if fmt == "bgr24" else ["-vf", "scale=512:512"]) + ["-f", "rawvideo", "-pix_fmt", fmt, "-"], capture_output=True, check=True).stdout
    return np.frombuffer(r, np.uint8).reshape(-1, 512, 512, ch) if ch > 1 else np.frombuffer(r, np.uint8).reshape(-1, 512, 512)
F = rd(a.layer); S = rd(a.src, "gray", 1); n = len(F); m = min(n, len(S))
G = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in F]
lo = (slice(280, 460), slice(140, 372))
jk = np.array([np.abs(G[t + 1][lo].astype(np.float32) - G[t][lo]).mean() for t in range(n - 1)])
js = np.array([np.abs(S[t + 1][lo].astype(np.float32) - S[t][lo]).mean() if t + 1 < m else np.nan for t in range(n - 1)]); ex = jk - js
runs = [(c["a"], min(c["b"], n)) for c in json.load(open(a.chunks))["chunks"]] if a.chunks else [(0, n)]
DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); GY, GX = np.mgrid[0:512, 0:512].astype(np.float32)
def interp(A, B, gA, gB, tau):   # backward-warp approximation of the in-between at tau (0 = A, 1 = B)
    fab = DIS.calc(gA, gB, None); fba = DIS.calc(gB, gA, None)
    wa = cv2.remap(A, GX - tau * fab[..., 0], GY - tau * fab[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    wb = cv2.remap(B, GX - (1 - tau) * fba[..., 0], GY - (1 - tau) * fba[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    return np.clip((1 - tau) * wa + tau * wb + 0.5, 0, 255).astype(np.uint8)
out = F.copy(); info = []
for r0, r1 in runs:
    if r1 - r0 < a.min_run: info.append({"run": [r0, r1], "skipped": "short"}); continue
    tt = np.arange(r0, min(r1, n) - 1); e_ = ex[tt]; ok = np.isfinite(e_); ph_ = (tt - r0) % a.period
    ph = np.array([e_[ok & (ph_ == q)].mean() if (ok & (ph_ == q)).any() else np.nan for q in range(a.period)])
    q = int(np.nanargmax(ph)) if a.force_phase < 0 else a.force_phase
    rest = float(np.nanmedian(np.delete(ph, q))); r = float(ph[q] / max(rest, 0.05)); nb = 0; wgt = 0.0
    wgt = 1.0 if a.force_phase >= 0 else float(np.clip((r - a.ratio) / max(a.ratio_full - a.ratio, 1e-6), 0, 1))
    if wgt > 0:
        for t in range(r0 + q, min(r1, n) - 1, a.period):   # the jump t -> t+1 (both frames inside this run); blended with the original by the weight
            A, B = F[t], F[t + 1]
            for u, tau, X in ((t, 1 / 3, A), (t + 1, 2 / 3, B)):
                I_ = interp(A, B, G[t], G[t + 1], tau).astype(np.float32)
                out[u] = np.clip((1 - wgt) * X.astype(np.float32) + wgt * I_ + 0.5, 0, 255).astype(np.uint8)
            nb += 1
    info.append({"run": [r0, r1], "phase": q, "peak_excess": round(float(ph[q]), 3), "rest_median": round(rest, 3), "ratio": round(r, 2), "weight": round(wgt, 3), "boundaries_fixed": nb})
p_ = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "512x512", "-r", "25", "-i", "-", "-c:v", "libx264", "-crf", "10"] + ks_cms.enc() + [a.out], stdin=subprocess.PIPE)
p_.stdin.write(out.tobytes()); p_.stdin.close(); p_.wait(); json.dump(info, open(a.out + ".seam.json", "w"))
for x in info: print("== ks_seam", json.dumps(x))
print(f"== ks_seam {len(runs)} run(s), {sum(x.get('boundaries_fixed', 0) for x in info)} boundaries smoothed -> {a.out}")
