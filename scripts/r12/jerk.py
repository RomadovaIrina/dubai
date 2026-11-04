#!/usr/bin/env python3
"""r13 LS: face jitter of a final vs the source (user on m03 LS_RATE 2/3: "лицо подергивается и дрожит"; the gate does not see it).
MediaPipe on the final (cached next to it); per source frame k the displacement d[k] = final landmarks - source landmarks (the upper face is the
source's own pixels, so head motion cancels; frames where the two faces disagree on the nose / eyes by > 0.15 IOD are dropped: the other face in
m03). Per group (jaw, mouth, cheeks): J[k] = |d[k+1] - 2 d[k] + d[k-1]| / IOD (point mean). Printed: J mean / p95 (x1000), the rate of J > --thr,
and the share of the d spectrum (2-15 Hz) at the 25/3 Hz cadence of LS_RATE 2/3 (+- 0.6 Hz).
    python scripts/r12/jerk.py --video FINAL.mp4 --src-lm <id>_source_lm_F.npz --label x [--json out.json]"""
import argparse, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "quality"))
import clean_face_stack as cfs   # noqa: E402

G = {"jaw": [288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58],
     "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
     "cheeks": [50, 280, 205, 425, 187, 411, 207, 427]}
RIG = [33, 133, 362, 263, 168, 6, 197, 4]
ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--src-lm", required=True); ap.add_argument("--label", default="")
ap.add_argument("--thr", type=float, default=0.01); ap.add_argument("--json", default=""); a = ap.parse_args()
v = pathlib.Path(a.video); lo = np.load(cfs.run_mediapipe(v, v.with_suffix(".jerk_mp.npz")))["lm"].astype(np.float64); ls = np.load(a.src_lm)["lm"].astype(np.float64)
import cv2   # noqa: E402
fps = cv2.VideoCapture(str(v)).get(cv2.CAP_PROP_FPS) or 30.0
n = min(len(lo), len(ls)); lo, ls = lo[:n, :, :2], ls[:n, :, :2]
iod = np.linalg.norm(ls[:, 33] - ls[:, 263], axis=1)
ok = np.isfinite(lo).all((1, 2)) & np.isfinite(ls).all((1, 2)) & (iod > 5)
ok &= np.nan_to_num(np.median(np.linalg.norm(lo[:, RIG] - ls[:, RIG], axis=2), 1) / np.maximum(iod, 1), nan=9) < 0.15
res = {"label": a.label, "frames": int(n), "valid": int(ok.sum()), "fps": round(fps, 2)}
for g, ix in G.items():
    d = (lo[:, ix] - ls[:, ix]) / iod[:, None, None]
    J = np.full(n, np.nan); v3 = ok[:-2] & ok[1:-1] & ok[2:]
    J[1:-1] = np.where(v3, np.linalg.norm(d[2:] - 2 * d[1:-1] + d[:-2], axis=2).mean(1), np.nan)
    Jv = J[np.isfinite(J)]
    # cadence share: the spectrum of the group-mean displacement (x and y) over the longest valid run
    s = d.mean(1); best = (0, 0); k = 0
    while k < n:
        if not ok[k]: k += 1; continue
        e = k
        while e < n and ok[e]: e += 1
        if e - k > best[1] - best[0]: best = (k, e)
        k = e
    seg = s[best[0]:best[1]]; share = None
    if len(seg) >= 64:
        seg = seg - seg.mean(0); P = (np.abs(np.fft.rfft(seg * np.hanning(len(seg))[:, None], axis=0)) ** 2).sum(1); f = np.fft.rfftfreq(len(seg), 1 / fps)
        band = (f >= 2) & (f <= 15); cad = band & (np.abs(f - 25 / 3) <= 0.6); share = round(float(P[cad].sum() / max(P[band].sum(), 1e-12)), 3)
    res[g] = {"J_mean_x1000": round(1000 * float(Jv.mean()), 2), "J_p95_x1000": round(1000 * float(np.percentile(Jv, 95)), 2),
              "rate_gt_thr": round(float((Jv > a.thr).mean()), 3), "cadence_share": share}
print("== jerk", a.label, json.dumps({k: res[k] for k in ("valid", "jaw", "mouth", "cheeks")}), flush=True)
if a.json: pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
