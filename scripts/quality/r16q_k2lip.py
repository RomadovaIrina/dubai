#!/usr/bin/env python3
"""r16 Q (from int scripts/keysync/r16/k2_lipmetrics.py, K2; audit r16 fix: the 12.5 Hz share is now taken on the SIGNED
frame-difference series dF - dS, not on |dF - dS|: abs() folds an every-other-frame shimmer (+a, -a, +a ...) into a constant -> DC, so the
old share_12.5Hz could not see the shimmer it was meant to find. mean / p95 / rate_gt1 stay on the absolute value (unchanged). The spectrum
uses only the longest run of consecutive valid frames (a face gap used to splice the series and break the time base).)
r16 K2 #11 / #12 on a FINAL (25 fps master grid) vs the master, the same definitions as the LatentSync work:
  #12 sticky lips = r9 artefacts.py (a) stick / unstick: o = inner-lip gap |13-14| / face height |10-152|, s[t] = o[t] - median(o[t-2..t+2]);
      an event when |s| > max(0.025, 4 x the source's MAD of s) and the source's own |s| at t is < half of it; runs merged (gap <= 1).
      Landmarks: the generated ones (KeySync raw 512, frame t - shift) vs the source crop's (same 512 grid). Also the source's own rate at the
      same threshold (how often she does that herself).
  #11 lip flicker = r14 tone_flicker.py on the master grid: per frame |dF - dS| of the mean grey of the lips hull / the lip ring (+0.3 IOD) /
      the low face; mean, p95, rate > 1 grey level, share of the spectrum at 12.5 Hz (an every-other-frame shimmer) and 5 Hz.
    python k2_lipmetrics.py --final F.mp4 --master M.mp4 --lm source_lm_F.npz --src-lm512 source_crop512.mediapipe.npz --ks-lm KS.npz --shift 1 --label x"""
import argparse, json
import numpy as np, cv2
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LOW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 1]
ap = argparse.ArgumentParser()
for k in ("final", "master", "lm", "src_lm512", "ks_lm"): ap.add_argument("--" + k.replace("_", "-"), required=True, dest=k)
ap.add_argument("--shift", type=int, default=0); ap.add_argument("--label", default=""); ap.add_argument("--t0", type=float, default=0); ap.add_argument("--t1", type=float, default=1e9)
a = ap.parse_args()
z = np.load(a.lm); LM = z["lm"][..., :2].astype(np.float64); J = z["j"]
SL = np.load(a.src_lm512)["lm"][..., :2].astype(np.float64); KL = np.load(a.ks_lm)["lm"][..., :2].astype(np.float64)
n = len(J); fps = 25.0


def op(p):
    if not np.isfinite(p[[13, 14, 10, 152]]).all(): return np.nan
    return np.linalg.norm(p[13] - p[14]) / max(np.linalg.norm(p[10] - p[152]), 1)


oS = np.array([op(SL[min(k, len(SL) - 1)]) for k in range(n)])
oF = np.array([op(KL[k - a.shift]) if 0 <= k - a.shift < len(KL) else np.nan for k in range(n)])


def spike(o):
    s = np.full(len(o), np.nan)
    for t in range(2, len(o) - 2):
        w = o[t - 2:t + 3]
        if np.isfinite(w).all(): s[t] = o[t] - np.median(w)
    return s


sS, sF = spike(oS), spike(oF); fin = np.isfinite(sS)
mad = float(np.median(np.abs(sS[fin] - np.median(sS[fin])))); thr = max(0.025, 4 * mad)
tw = (np.arange(n) / fps >= a.t0) & (np.arange(n) / fps <= a.t1)


def events(flag):
    idx = np.where(flag & tw)[0]; ev = []
    for i in idx:
        if ev and i - ev[-1][1] <= 2: ev[-1][1] = i
        else: ev.append([i, i])
    return ev


evF = events(np.nan_to_num(np.abs(sF)) > thr) ; evF = [e for e in evF if np.nanmin(np.abs(sS[e[0]:e[1] + 1])) < thr / 2]
evS = events(np.nan_to_num(np.abs(sS)) > thr)
dur = tw.sum() / fps / 60
res = {"label": a.label, "minutes": round(dur, 2), "stick_thr": round(thr, 4),
       "stick_events": len(evF), "stick_per_min": round(len(evF) / dur, 2), "src_own_spikes_per_min": round(len(evS) / dur, 2),
       "stick_times": [round(e[0] / fps, 2) for e in evF][:20]}
# ---- lip flicker on the master grid
cv, cs = cv2.VideoCapture(a.final), cv2.VideoCapture(a.master); R = {r: [] for r in ("lips", "ring", "low")}; prev = None
for k in range(n):
    okv, fv = cv.read(); oks, fs = cs.read()
    if not (okv and oks): break
    p = LM[min(int(J[k]), len(LM) - 1)]
    if not (tw[k] and np.isfinite(p[LIPS + LOW + [33, 263]]).all()): prev = None; continue
    iod = float(np.linalg.norm(p[33] - p[263])); H, W = fs.shape[:2]
    hl = np.zeros((H, W), np.uint8); cv2.fillPoly(hl, [np.round(cv2.convexHull(p[LIPS].astype(np.float32))[:, 0]).astype(np.int32)], 1)
    dl = cv2.dilate(hl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.3 * iod) + 1,) * 2))
    lo = np.zeros((H, W), np.uint8); cv2.fillPoly(lo, [np.round(cv2.convexHull(p[LOW].astype(np.float32))[:, 0]).astype(np.int32)], 1)
    m = {"lips": hl > 0, "ring": (dl > 0) & (hl == 0), "low": (lo > 0) & (dl == 0)}
    gv, gs = cv2.cvtColor(fv, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.cvtColor(fs, cv2.COLOR_BGR2GRAY).astype(np.float32)
    cur = {r: (float(gv[mm].mean()), float(gs[mm].mean())) for r, mm in m.items()}
    if prev is not None:
        for r in R: R[r].append((cur[r][0] - prev[r][0]) - (cur[r][1] - prev[r][1]))
    else:
        for r in R: R[r].append(np.nan)   # a run break: no splice across a gap
    prev = cur
for r, v in R.items():
    v = np.array(v, np.float64); fin = np.isfinite(v)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], fin.astype(int), [0]])))   # runs of valid frames
    r0, r1 = max(zip(edges[::2], edges[1::2]), key=lambda e: e[1] - e[0]) if len(edges) else (0, 0)
    sv = v[r0:r1]; v = np.abs(v[fin])
    f = np.fft.rfftfreq(len(sv), 1 / fps); P = np.abs(np.fft.rfft((sv - sv.mean()) * np.hanning(len(sv)))) ** 2; band = (f >= 0.5) & (f <= 12.5)
    sh = lambda hz: round(float(P[band & (np.abs(f - hz) <= 0.3)].sum() / max(P[band].sum(), 1e-12)), 3)
    res["flicker_" + r] = {"mean": round(float(v.mean()), 3), "p95": round(float(np.percentile(v, 95)), 3), "rate_gt1": round(float((v > 1).mean()), 3), "share_12.5Hz": sh(12.5), "share_5Hz": sh(5.0), "fft_frames": int(len(sv))}
print("K2_LIPM " + json.dumps(res), flush=True)
