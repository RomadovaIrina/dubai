#!/usr/bin/env python3
"""r13 LS: pixel-level face jitter of a final vs the source (landmarks miss it: MediaPipe's video mode smooths over time).
Per source frame pair (k, k+1), on the lower face (jaw / cheeks / chin box from the source landmarks, the lips hull + 0.15 IOD excluded):
  R = DIS flow(final k -> k+1) - DIS flow(source k -> k+1)   (the motion the final has and the source does not)
  shake[k]  = |median R| / IOD                      (the pasted region sliding as a whole)
  wobble[k] = mean |R - median R| / IOD             (local deformation: trembling texture / edges)
  flick[k]  = mean |grey final - grey source| change between k and k+1 (tone flicker of the pasted region)
Printed: means (x1000), p95, and the share of each series' 2-15 Hz spectrum at the 25/3 Hz cadence of LS_RATE 2/3 (+- 0.6 Hz).
    python scripts/r12/wobble.py --video FINAL.mp4 --src SOURCE.mp4 --src-lm <id>_source_lm_F.npz --label x [--json out.json] [--max-frames N]"""
import argparse, json, pathlib
import numpy as np, cv2

LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LOW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 1]
ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--src", required=True); ap.add_argument("--src-lm", required=True)
ap.add_argument("--label", default=""); ap.add_argument("--json", default=""); ap.add_argument("--max-frames", type=int, default=0); a = ap.parse_args()
lm = np.load(a.src_lm)["lm"].astype(np.float64)
cv, cs = cv2.VideoCapture(a.video), cv2.VideoCapture(a.src); fps = cv.get(cv2.CAP_PROP_FPS) or 30.0
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
prev = None; sh, wb, fl = [], [], []; k = 0
while True:
    okv, fv = cv.read(); oks, fs = cs.read()
    if not (okv and oks) or (a.max_frames and k >= a.max_frames): break
    p = lm[k] if k < len(lm) else None
    if p is None or not np.isfinite(p[LOW + LIPS + [33, 263]]).all():
        prev = None; sh.append(np.nan); wb.append(np.nan); fl.append(np.nan); k += 1; continue
    iod = float(np.linalg.norm(p[33, :2] - p[263, :2])); q = p[LOW, :2]
    x0, y0 = np.maximum(q.min(0).astype(int) - 4, 0); x1, y1 = q.max(0).astype(int) + 4
    s = max(1.0, iod / 60.0)   # work at ~60 px IOD
    crop = lambda f: cv2.resize(cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA)
    gv, gs = crop(fv), crop(fs)
    m = np.zeros(gv.shape, np.uint8); cv2.fillPoly(m, [np.round((cv2.convexHull(q.astype(np.float32))[:, 0] - [x0, y0]) / s).astype(np.int32)], 1)
    hl = np.zeros(gv.shape, np.uint8); cv2.fillPoly(hl, [np.round((cv2.convexHull(p[LIPS, :2].astype(np.float32))[:, 0] - [x0, y0]) / s).astype(np.int32)], 1)
    hl = cv2.dilate(hl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.15 * iod / s) + 1,) * 2)); m = (m > 0) & (hl == 0)
    if prev is not None and prev[0].shape == gv.shape:
        Fv = dis.calc(prev[0], gv, None); Fs = dis.calc(prev[1], gs, None); R = (Fv - Fs)[prev[2] & m]
        if len(R) > 50:
            med = np.median(R, 0); sh.append(float(np.linalg.norm(med)) * s / iod); wb.append(float(np.linalg.norm(R - med, axis=1).mean()) * s / iod)
            dv = gv.astype(np.float32) - gs.astype(np.float32); dp = prev[0].astype(np.float32) - prev[1].astype(np.float32); fl.append(float(np.abs(dv - dp)[prev[2] & m].mean()))
        else: sh.append(np.nan); wb.append(np.nan); fl.append(np.nan)
    else: sh.append(np.nan); wb.append(np.nan); fl.append(np.nan)
    prev = (gv, gs, m); k += 1
res = {"label": a.label, "frames": k, "fps": round(fps, 2)}
for name, x in (("shake", sh), ("wobble", wb), ("flicker", fl)):
    x = np.asarray(x, float); v = x[np.isfinite(x)]; xs = np.where(np.isfinite(x), x, np.nanmean(x)); xs = xs - xs.mean()
    P = np.abs(np.fft.rfft(xs * np.hanning(len(xs)))) ** 2; f = np.fft.rfftfreq(len(xs), 1 / fps); band = (f >= 2) & (f <= 15); cad = band & (np.abs(f - 25 / 3) <= 0.6)
    sc = 1 if name == "flicker" else 1000
    res[name] = {"mean": round(sc * float(v.mean()), 3), "p95": round(sc * float(np.percentile(v, 95)), 3), "cadence_share": round(float(P[cad].sum() / max(P[band].sum(), 1e-12)), 3)}
print("== wobble", a.label, json.dumps({k_: res[k_] for k_ in ("shake", "wobble", "flicker")}), flush=True)
if a.json: pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
