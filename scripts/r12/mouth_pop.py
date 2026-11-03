#!/usr/bin/env python3
"""r13 LS: temporal irregularity of the MOUTH of a final (no source reference: the mouth is generated).
Per frame k: frames k-1 and k+1 flow-warped onto k (DIS, grey, mouth box), pop[k] = mean |f_k - (w_{k-1} + w_{k+1}) / 2| over the lips hull
+ --dil IOD, and the same for the source (pop_src). A smooth motion leaves ~0; a tremble / snap / sharpness pulse leaves a residual.
Printed: mean / p95 of pop, the excess over the source, the rate of frames with pop > --thr, and the 25/3 Hz cadence share of the series.
    python scripts/r12/mouth_pop.py --video FINAL.mp4 --src SOURCE.mp4 --src-lm <id>_source_lm_F.npz --label x"""
import argparse, json, pathlib
import numpy as np, cv2
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--src", required=True); ap.add_argument("--src-lm", required=True)
ap.add_argument("--label", default=""); ap.add_argument("--dil", type=float, default=0.3); ap.add_argument("--thr", type=float, default=4.0); ap.add_argument("--json", default="")
a = ap.parse_args(); lm = np.load(a.src_lm)["lm"].astype(np.float64); dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def frames(p):
    c = cv2.VideoCapture(p); out = []
    while True:
        ok, f = c.read()
        if not ok: break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    return out, c.get(cv2.CAP_PROP_FPS) or 30.0


def series(F):
    n = min(len(F), len(lm)); pop = np.full(n, np.nan)
    for k in range(1, n - 1):
        p = lm[k]
        if not np.isfinite(p[LIPS + [33, 263]]).all(): continue
        iod = float(np.linalg.norm(p[33, :2] - p[263, :2])); q = p[LIPS, :2]; r = int(0.6 * iod)
        x0, y0 = np.maximum(q.min(0).astype(int) - r, 0); x1, y1 = q.max(0).astype(int) + r
        A, B, C = [np.ascontiguousarray(F[j][y0:y1, x0:x1]) for j in (k - 1, k, k + 1)]
        if B.size < 400: continue
        gy, gx = np.mgrid[0:B.shape[0], 0:B.shape[1]].astype(np.float32)
        wa = cv2.remap(A, gx + (f := dis.calc(B, A, None))[..., 0], gy + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        wc = cv2.remap(C, gx + (g := dis.calc(B, C, None))[..., 0], gy + g[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        m = np.zeros(B.shape, np.uint8); cv2.fillPoly(m, [np.round(cv2.convexHull(q.astype(np.float32))[:, 0] - [x0, y0]).astype(np.int32)], 1)
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(a.dil * iod) + 1,) * 2)) > 0
        pop[k] = float(np.abs(B.astype(np.float32) - 0.5 * (wa + wc))[m].mean())
    return pop


Fv, fps = frames(a.video); Fs, _ = frames(a.src); pv, ps = series(Fv), series(Fs); n = min(len(pv), len(ps)); pv, ps = pv[:n], ps[:n]
ok = np.isfinite(pv) & np.isfinite(ps); x = pv - ps
xs = np.where(ok, pv, np.nanmean(pv)); xs = xs - xs.mean(); P = np.abs(np.fft.rfft(xs * np.hanning(len(xs)))) ** 2; f = np.fft.rfftfreq(len(xs), 1 / fps)
band = (f >= 2) & (f <= 15); cad = band & (np.abs(f - 25 / 3) <= 0.6)
res = {"label": a.label, "pop_mean": round(float(pv[ok].mean()), 3), "pop_p95": round(float(np.percentile(pv[ok], 95)), 3), "src_pop_mean": round(float(ps[ok].mean()), 3),
       "excess_mean": round(float(x[ok].mean()), 3), "rate_gt_thr": round(float((pv[ok] > a.thr).mean()), 3), "cadence_share": round(float(P[cad].sum() / max(P[band].sum(), 1e-12)), 3)}
print("== mouth_pop", json.dumps(res), flush=True)
if a.json: pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
