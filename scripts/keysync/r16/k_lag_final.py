#!/usr/bin/env python3
"""r16 K1: head lag of the PASTED final vs the source on the master grid. Anchor = eye corners + nose bridge (33, 263, 133, 362, 168).
Final landmarks: mediapipe on the final (master grid, frame k). Source: source_lm_F.npz lm[j[k]]. Prints the best velocity lag (frames, + = final is late),
corr at lag 0, the mean / p95 anchor offset |final - source| px (the visible "mask lags the head"), and the same over the fast-head frames (top 20 % speed).
    python k_lag_final.py --final FINAL.mediapipe.npz --src source_lm_F.npz [--label x]"""
import argparse, json
import numpy as np
AN = [33, 263, 133, 362, 168]
ap = argparse.ArgumentParser(); ap.add_argument("--final", required=True); ap.add_argument("--src", required=True); ap.add_argument("--label", default=""); a = ap.parse_args()
F = np.load(a.final)["lm"][..., :2].astype(np.float64); z = np.load(a.src); L = z["lm"][..., :2].astype(np.float64); J = z["j"] if "j" in z.files else np.arange(len(L))
if F.max() < 2: F = F * np.array([float(x) for x in np.load(a.final)["size"]][::-1]) if "size" in np.load(a.final).files else F
n = min(len(F), len(J)); S = L[np.minimum(J[:n], len(L) - 1)]
fa, sa = F[:n, AN].mean(1), S[:, AN].mean(1); ok = np.isfinite(fa).all(1) & np.isfinite(sa).all(1)
for x in (fa, sa):
    for d in (0, 1): x[~ok, d] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), x[ok, d])
vs, vf = np.diff(sa, axis=0), np.diff(fa, axis=0)
def corr(lag):
    x, y = (vs[: len(vs) - lag], vf[lag:]) if lag >= 0 else (vs[-lag:], vf[: len(vf) + lag])
    return float(np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y)))
cs = {l: corr(l) for l in range(-4, 5)}; best = max(cs, key=cs.get)
off = np.linalg.norm(fa - sa, axis=1)[1:]; sp = np.linalg.norm(vs, axis=1); fast = sp >= np.percentile(sp, 80)
print(json.dumps({"label": a.label, "frames": int(n), "best_lag": best, "corr_best": round(cs[best], 3), "corr_lag0": round(cs[0], 3), "corr_lag1": round(cs[1], 3),
  "offset_mean_px": round(float(off.mean()), 2), "offset_p95_px": round(float(np.percentile(off, 95)), 2), "offset_fast_mean_px": round(float(off[fast].mean()), 2),
  "head_speed_mean": round(float(sp.mean()), 2)}))
