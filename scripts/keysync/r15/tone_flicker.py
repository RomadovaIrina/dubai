#!/usr/bin/env python3
"""r14 F: lower-face / mouth BRIGHTNESS flicker of a final vs the source on the source frame grid (user on old03: "the lower face blinks").
Per source frame k: dS[k] = mean grey(k) - mean grey(k-1) of the source, dF[k] the same of the final, over three regions from the source landmarks:
  low = jaw / cheeks with the lips + 0.3 IOD excluded; ring = the lip ring (lips hull + 0.3 IOD minus the hull); lips = the lip hull.
flicker[k] = |dF - dS| (the final's own brightness change). Printed per region: mean, p95, the rate > --thr (grey levels), the top frames
(with time), and the share of the 2-15 Hz spectrum at 5 Hz (the 25 -> 30 retime cycle) and 25/16 Hz (the 16-frame LatentSync window).
    python scripts/r12/tone_flicker.py --video FINAL.mp4 --src SOURCE --src-lm <id>_source_lm_F.npz --label x [--t0 S --t1 S]"""
import argparse, json
import numpy as np, cv2
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LOW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 1]
ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--src", required=True); ap.add_argument("--src-lm", required=True)
ap.add_argument("--label", default=""); ap.add_argument("--thr", type=float, default=1.0); ap.add_argument("--t0", type=float, default=0); ap.add_argument("--t1", type=float, default=1e9)
a = ap.parse_args(); lm = np.load(a.src_lm)["lm"].astype(np.float64)
cv, cs = cv2.VideoCapture(a.video), cv2.VideoCapture(a.src); fps = cs.get(cv2.CAP_PROP_FPS) or 30.0
R = {r: [] for r in ("low", "ring", "lips")}; ks = []; prev = None; k = 0
while True:
    okv, fv = cv.read(); oks, fs = cs.read()
    if not (okv and oks): break
    t = k / fps
    if a.t0 <= t <= a.t1 and k < len(lm) and np.isfinite(lm[k][LIPS + LOW + [33, 263]]).all():
        p = lm[k]; iod = float(np.linalg.norm(p[33, :2] - p[263, :2])); H, W = fs.shape[:2]
        hl = np.zeros((H, W), np.uint8); cv2.fillPoly(hl, [np.round(cv2.convexHull(p[LIPS, :2].astype(np.float32))[:, 0]).astype(np.int32)], 1)
        dl = cv2.dilate(hl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.3 * iod) + 1,) * 2))
        lo = np.zeros((H, W), np.uint8); cv2.fillPoly(lo, [np.round(cv2.convexHull(p[LOW, :2].astype(np.float32))[:, 0]).astype(np.int32)], 1)
        m = {"low": (lo > 0) & (dl == 0), "ring": (dl > 0) & (hl == 0), "lips": hl > 0}
        gv, gs = cv2.cvtColor(fv, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.cvtColor(fs, cv2.COLOR_BGR2GRAY).astype(np.float32)
        cur = {r: (float(gv[mm].mean()), float(gs[mm].mean())) for r, mm in m.items()}
        if prev is not None and prev[0] == k - 1:
            for r in R: R[r].append(abs((cur[r][0] - prev[1][r][0]) - (cur[r][1] - prev[1][r][1])))
            ks.append(k)
        prev = (k, cur)
    k += 1
res = {"label": a.label, "frames": len(ks)}
for r, v in R.items():
    v = np.array(v); f = np.fft.rfftfreq(len(v), 1 / fps); P = np.abs(np.fft.rfft((v - v.mean()) * np.hanning(len(v)))) ** 2; band = (f >= 0.5) & (f <= 15)
    sh = lambda hz: round(float(P[band & (np.abs(f - hz) <= 0.3)].sum() / max(P[band].sum(), 1e-12)), 3)
    top = np.argsort(-v)[:6]
    res[r] = {"mean": round(float(v.mean()), 3), "p95": round(float(np.percentile(v, 95)), 3), "rate_gt_thr": round(float((v > a.thr).mean()), 3),
              "share_5Hz": sh(5.0), "share_win16": sh(25 / 16), "top": [[round(ks[i] / fps, 2), round(float(v[i]), 2)] for i in sorted(top)]}
print("== tone_flicker", json.dumps(res), flush=True)
