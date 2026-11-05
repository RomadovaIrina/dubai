#!/usr/bin/env python3
"""r15 F: does the rigid head motion H (master_rigid.npz, DIS flow + RANSAC on the upper face) disagree with the landmarks, and how far does the
flow:S anchor filter (LatentSync's crop / paste-back geometry) then drift from the raw landmarks? With --fix: H[t] is replaced by the
landmark similarity (rigid points t-1 -> t) where the two disagree by more than --thr IOD (anchor_filters.validate_motion).
    python scripts/r15/anchor_check.py <run> [--thr 0.02] [--sigma 4]"""
import sys, pathlib, argparse
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot")); import anchor_filters as af
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--thr", type=float, default=0.02); ap.add_argument("--sigma", type=float, default=4)
a = ap.parse_args(); W = f"/workspace/dub/artifacts/r4m/r9g_{a.run}/work_{a.run}_r9g_{a.run}/video"
H = np.load(W + "/master_rigid.npz")["H"]; lm = np.load(W + "/master_mediapipe.npz")["lm"][:, :, :2].astype(np.float32); n = min(len(H), len(lm)); H, lm = H[:n], lm[:n]
iod = np.linalg.norm(lm[:, 33] - lm[:, 263], axis=1)
Hf, info = af.validate_motion(H, lm, a.thr, return_info=True)
PTS = [1, 33, 263, 61, 291, 152]   # what the crop / affine follows (eyes, nose, mouth corners, chin)
for lab, HH in (("original H", H), ("validated H", Hf)):
    f = af.filter_landmarks(lm, HH, f"flow:{a.sigma:g}"); d = np.linalg.norm((f[:, PTS] - lm[:, PTS]).mean(1), axis=1) / iod
    bad = np.flatnonzero(np.nan_to_num(d) > 0.02)
    print(f"== {a.run} {lab}: anchor drift |flow:{a.sigma:g} - raw| (IOD) p50 {np.nanpercentile(d, 50):.4f} p99 {np.nanpercentile(d, 99):.4f} max {np.nanmax(d):.3f}; frames > 0.02: {len(bad)} {bad.tolist()[:30]}", flush=True)
    if a.run == "q2_m03": print("   466-486:", " ".join(f"{t}:{d[t]:.3f}" for t in range(466, 487)), flush=True)
print(f"== {a.run} validate_motion: {info}", flush=True)
