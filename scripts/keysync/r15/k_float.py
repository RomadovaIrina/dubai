#!/usr/bin/env python3
"""r15 K: which composite stage makes the lower face "float"? MediaPipe landmarks on each stage's video; per frame the cheek (234 / 454), the mouth
corners (61 / 291) and the chin (152) relative to the eye anchor (133 / 362 midpoint), in IOD; jerk = |second difference| * fps^2 (IOD / s^2, so
25 and 30 fps stages compare); mean and p95 per window (default the user's m03 spots 2-4 / 6-8 / 11-13 s) and overall. Same metric on the source.
    python k_float.py --videos label=path ... --windows 2-4 6-8 11-13 --json out.json   (landmarks cached next to each video as <path>.flt.npz)"""
import argparse, json, os, subprocess
import numpy as np, cv2

PTS = {"cheek_l": [234], "cheek_r": [454], "mouth_corners": [61, 291], "chin": [152]}
ap = argparse.ArgumentParser(); ap.add_argument("--videos", nargs="+", required=True); ap.add_argument("--windows", nargs="+", default=["2-4", "6-8", "11-13"])
ap.add_argument("--json", required=True); a = ap.parse_args()
V = [v.split("=", 1) for v in a.videos]; procs = []
for lb, p in V:
    npz = p + ".flt.npz"
    if not os.path.exists(npz):
        procs.append(subprocess.Popen(["/venv/mediapipe/bin/python", "/workspace/dub/dubai/scripts/quality/mp_landmarks.py", "--video", p, "--out", npz], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
for pr in procs: pr.wait()
out = {}
for lb, p in V:
    lm = np.load(p + ".flt.npz")["lm"][..., :2].astype(np.float64); fps = cv2.VideoCapture(p).get(cv2.CAP_PROP_FPS) or 25.0
    an = (lm[:, 133] + lm[:, 362]) / 2; iod = np.linalg.norm(lm[:, 33] - lm[:, 263], axis=1)[:, None]; t = np.arange(len(lm)) / fps; r = {"fps": round(fps, 2)}
    for k, idx in PTS.items():
        x = (lm[:, idx].mean(1) - an) / iod; j = np.full(len(x), np.nan); j[1:-1] = np.linalg.norm(x[1:-1] - (x[:-2] + x[2:]) / 2, axis=1) * fps * fps
        r[k] = {"all": round(float(np.nanmean(j)), 2)}
        for w in a.windows:
            t0, t1 = [float(z) for z in w.split("-")]; s = j[(t >= t0) & (t <= t1)]; s = s[np.isfinite(s)]
            r[k][w] = [round(float(s.mean()), 2), round(float(np.percentile(s, 95)), 2)] if len(s) else None
    out[lb] = r
json.dump(out, open(a.json, "w"), indent=1)
for k in PTS:
    print(f"{k:14s} " + " | ".join(f"{lb} {out[lb][k]['all']:.2f} [" + " ".join(f"{out[lb][k][w][0]:.1f}" for w in a.windows) + "]" for lb, _ in V))
