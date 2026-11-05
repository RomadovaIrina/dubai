#!/usr/bin/env python3
"""r15 K: does the KeySync crop's head lag the source? Head track = the eye-corner / nose-bridge anchor (33, 263, 133, 362, 168) of mediapipe landmarks on
the SAME 512 crop grid (source crop vs raw KeySync output, frame i = frame i). Velocity cross-correlation over lags -8..+8 frames (25 fps): the best lag,
its correlation, the amplitude ratio KS / source, and the mean anchor offset |KS - source| in px; plus the head speed of the source (px / frame).
A positive lag = KeySync is LATE. The plain paste adds its AV shift on top (KeySync frame i shown at master frame i + shift).
    python k_lag_head.py --src SRC_crop.npz --ks KS.npz [--label x]"""
import argparse, json
import numpy as np

AN = [33, 263, 133, 362, 168]
ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--ks", required=True); ap.add_argument("--label", default=""); a = ap.parse_args()
S = np.load(a.src)["lm"][..., :2].astype(np.float64); K = np.load(a.ks)["lm"][..., :2].astype(np.float64); n = min(len(S), len(K))
sa, ka = S[:n, AN].mean(1), K[:n, AN].mean(1); ok = np.isfinite(sa).all(1) & np.isfinite(ka).all(1)
for x in (sa, ka):   # fill gaps by interpolation (velocity needs continuity)
    for d in (0, 1): x[~ok, d] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), x[ok, d])
vs, vk = np.diff(sa, axis=0), np.diff(ka, axis=0)
best = None
for lag in range(-8, 9):   # vk[t] ~ vs[t - lag]
    if lag >= 0: x, y = vs[: len(vs) - lag], vk[lag:]
    else: x, y = vs[-lag:], vk[: len(vk) + lag]
    c = float(np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if best is None or c > best[1]: best = (lag, c)
out = {"label": a.label, "frames": int(n), "best_lag_frames": best[0], "corr": round(best[1], 3), "corr_lag0": None,
       "amp_ratio_ks_over_src": round(float(np.linalg.norm(vk, axis=1).mean() / max(np.linalg.norm(vs, axis=1).mean(), 1e-9)), 3),
       "mean_anchor_offset_px": round(float(np.linalg.norm(ka - sa, axis=1).mean()), 2), "src_head_speed_px_per_frame": round(float(np.linalg.norm(vs, axis=1).mean()), 2),
       "src_head_speed_p95": round(float(np.percentile(np.linalg.norm(vs, axis=1), 95)), 2)}
out["corr_lag0"] = round(float(np.sum(vs * vk) / np.sqrt(np.sum(vs * vs) * np.sum(vk * vk))), 3)
print(json.dumps(out))
