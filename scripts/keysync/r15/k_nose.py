#!/usr/bin/env python3
"""r15 K: nose-tip / upper-face tremble of a KeySync layer vs the source crop (the user on m18 fast4: "the tip of the nose trembles a bit").
Both = mediapipe landmarks on the 512^2 / 25 fps grid (the source crop data/videos/<id>.mp4 and the raw KeySync output), frames aligned from 0.
Per frame, positions relative to a head anchor (the eye inner corners' midpoint, 133 / 362), in units of the inter-ocular distance (IOD):
  jitter = |second difference| (frame t vs the mean of t-1 / t+1) -> mean, p95, the rate > 0.01 IOD; for the nose tip (1), the nostrils (98 / 327),
  the eyes (33 / 263), the brows (105 / 334); dev = |layer - source| per frame (how far KeySync moves the point), and its own jitter.
    python k_nose.py --src SRC.npz --ks A.npz [B.npz ...] --labels a b ... [--fps 25]"""
import argparse, json
import numpy as np

PTS = {"nose_tip": [1], "nostrils": [98, 327], "eyes": [33, 263], "brows": [105, 334], "upper_lip": [0, 13]}


def load(f):
    lm = np.load(f)["lm"][..., :2].astype(np.float64)
    return lm


def rel(lm):
    a = (lm[:, 133] + lm[:, 362]) / 2; iod = np.linalg.norm(lm[:, 33] - lm[:, 263], axis=1)[:, None]
    return {k: (lm[:, v].mean(1) - a) / iod for k, v in PTS.items()}, a, iod


def jit(x):
    j = np.linalg.norm(x[1:-1] - (x[:-2] + x[2:]) / 2, axis=1); j = j[np.isfinite(j)]
    return {"mean": round(float(j.mean()) * 1000, 2), "p95": round(float(np.percentile(j, 95)) * 1000, 2), "rate_gt_10": round(float((j > 0.01).mean()), 4)}


ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--ks", nargs="+", required=True); ap.add_argument("--labels", nargs="+", required=True)
a = ap.parse_args(); S = load(a.src); rs, _, _ = rel(S); out = {"units": "1e-3 IOD per frame", "source": {k: jit(v) for k, v in rs.items()}}
for f, lb in zip(a.ks, a.labels):
    L = load(f); n = min(len(L), len(S)); rl, _, _ = rel(L[:n]); r = {}
    for k in PTS:
        d = rl[k] - rs[k][:n]; dv = np.linalg.norm(d, axis=1); dv = dv[np.isfinite(dv)]
        r[k] = {**jit(rl[k]), "dev_mean": round(float(dv.mean()) * 1000, 2), "dev_jitter_mean": jit(d)["mean"]}
    out[lb] = r
print(json.dumps(out))
for k in PTS:
    print(f"{k:10s} jitter mean (1e-3 IOD): src {out['source'][k]['mean']:6.2f} | " + " | ".join(f"{lb} {out[lb][k]['mean']:6.2f} (dev {out[lb][k]['dev_mean']:5.1f})" for lb in a.labels))
