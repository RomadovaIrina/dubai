#!/usr/bin/env python3
"""r20 twitch locator (Irina 15.11: "дерганья"): where and how the pasted face twitches against the source.
R = result - source (luma). The neighbours k-1 / k+1 are brought onto frame k with the SOURCE's own dense flow (Farneback; the same warp for the
result), so head / camera motion and the source texture cancel and no landmark jitter enters. J = mean |R[k+1] - 2 R[k] + R[k-1]| per region =
how fast the difference to the source jerks (codec noise only where the result is the source). D = mean |R[k]| = how far it is from the source.
Regions (source landmarks of frame k): face = inside the face oval below the nose tip, minus the mouth zone; neck = a band 0.45 iod below the
jaw line; mouth = the lip hull + 0.12 iod (reference only: lips differ by design); bg = a ring outside the face (the codec floor).
Prints the top peaks per region, 5 s means, the 13 / 14-frame phase profile (generator chunk borders).
    twitch.py --src ks/master_25fps.mp4 --res out.mp4 --lm ks/lm_f0.npz [--k0 0] [--json out.json] [--npz series.npz]"""
import argparse, json
import numpy as np, cv2
ap = argparse.ArgumentParser(); [ap.add_argument("--" + k, required=True) for k in ("src", "res", "lm")]
ap.add_argument("--k0", type=int, default=0, help="first master frame of --res (a shot cut)"); ap.add_argument("--json", default=""); ap.add_argument("--npz", default=""); ap.add_argument("--top", type=int, default=12); ap.add_argument("--lm-fps", type=float, default=0.0, help="the landmarks' frame rate when --src / --res run at another (native-fps finals: 25)"); ap.add_argument("--periods", default="13,14"); ap.add_argument("--iod", type=float, default=110.0, help="work scale: the face is resized to this inter-ocular distance")
a = ap.parse_args()
z = np.load(a.lm); LM = z["lm"][..., :2].astype(np.float32); J_ = z["j"] if "j" in z.files else np.arange(len(LM))
OVAL = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]; JAW = OVAL[3:-3]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
cs, cr = cv2.VideoCapture(a.src), cv2.VideoCapture(a.res); cs.set(cv2.CAP_PROP_POS_FRAMES, a.k0)
fps = cr.get(cv2.CAP_PROP_FPS) or 25.0; n = int(min(cs.get(cv2.CAP_PROP_FRAME_COUNT) - a.k0, cr.get(cv2.CAP_PROP_FRAME_COUNT), (len(J_) - a.k0) if not a.lm_fps else 1e9)); PER = [int(x) for x in a.periods.split(",")]
names = ["face", "neck", "mouth", "bg"]; NR = len(names); out = np.full((n, NR), np.nan, np.float32); amp = np.full((n, NR), np.nan, np.float32)
Mo = np.full((n, NR), np.nan, np.float32); Js = np.full((n, NR), np.nan, np.float32); Jr = np.full((n, NR), np.nan, np.float32); SH = np.full((n, 2), np.nan, np.float32)
def masks(p, shape, iod):
    m = np.zeros((NR,) + shape, np.uint8); cv2.fillConvexPoly(m[2], cv2.convexHull(p[LIPS].astype(np.int32)), 1)
    m[2] = cv2.dilate(m[2], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.12 * iod) + 1,) * 2))
    cv2.fillPoly(m[0], [p[OVAL].astype(np.int32)], 1); m[0][:max(0, int(p[4][1]))] = 0
    jw = p[JAW]; cv2.fillPoly(m[1], [np.concatenate([jw, jw[::-1] + np.float32([0, 0.45 * iod])]).astype(np.int32)], 1)
    big = cv2.dilate(np.maximum(m[0], m[1]), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.25 * iod) + 1,) * 2)); m[3] = 1 - big
    m[1][m[0] > 0] = 0; m[0][m[2] > 0] = 0; m[1][m[2] > 0] = 0
    return m.astype(bool)
def crop(f, box, sc):
    x0, y0, x1, y1 = box; c = cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(cv2.resize(c, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 0.999 else c, (3, 3), 0).astype(np.float32)
def warp(img, fl):
    h, w = img.shape; gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
FB = dict(pyr_scale=0.5, levels=4, winsize=21, iterations=3, poly_n=7, poly_sigma=1.5, flags=0)
buf = []   # (src frame, res frame) of k-1, k, k+1
def rd():
    ok1, fs = cs.read(); ok2, fr = cr.read()
    if not (ok1 and ok2): return None
    if fr.shape != fs.shape: fr = cv2.resize(fr, (fs.shape[1], fs.shape[0]), interpolation=cv2.INTER_AREA)
    return fs, fr
buf = [rd(), rd()]
for k in range(1, n - 1):
    buf.append(rd())
    if buf[-1] is None: n = k + 1; break
    kl = a.k0 + k if not a.lm_fps else int((a.k0 + k) * a.lm_fps / fps + 0.5); p = LM[min(int(J_[min(kl, len(J_) - 1)]), len(LM) - 1)]
    if np.isfinite(p[[33, 263, 4, 152]]).all():
        iod = float(np.linalg.norm(p[33] - p[263])); H, W = buf[1][0].shape[:2]; q = p[np.isfinite(p).all(1)]
        x0, y0 = np.maximum(q.min(0) - 0.6 * iod, 0).astype(int); x1, y1 = np.minimum(q.max(0) + 0.8 * iod, [W, H]).astype(int); sc = min(1.0, a.iod / max(iod, 1.0))
        if x1 - x0 > 16 and y1 - y0 > 16:
            box = (x0, y0, x1, y1); S = [crop(b[0], box, sc) for b in buf]; O = [crop(b[1], box, sc) for b in buf]; R = [O[1] - S[1]]; WS = [S[1]]; WO = [O[1]]
            u8 = lambda x: np.clip(x, 0, 255).astype(np.uint8)
            for j in (0, 2):
                fl = cv2.calcOpticalFlowFarneback(u8(S[1]), u8(S[j]), None, **FB); WO.append(warp(O[j], fl)); WS.append(warp(S[j], fl)); R.append(WO[-1] - WS[-1])
            d2 = np.abs(R[1] + R[2] - 2 * R[0]); M = masks((p - np.float32([x0, y0])) * sc, S[1].shape, iod * sc)
            ds = np.abs(WS[1] + WS[2] - 2 * WS[0]); do = np.abs(WO[1] + WO[2] - 2 * WO[0])
            for i in range(NR):
                if M[i].sum() > 50: out[k, i] = d2[M[i]].mean(); amp[k, i] = np.abs(R[0])[M[i]].mean(); Mo[k, i] = R[0][M[i]].mean(); Js[k, i] = ds[M[i]].mean(); Jr[k, i] = do[M[i]].mean()
            ys, xs = np.nonzero(M[0] | M[2])
            if len(ys) > 200:
                a0, a1, b0, b1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1; hw = cv2.createHanningWindow((int(b1 - b0), int(a1 - a0)), cv2.CV_32F)
                gr = lambda x: cv2.Laplacian(cv2.GaussianBlur(x[a0:a1, b0:b1], (0, 0), 1.5), cv2.CV_32F)
                (dx, dy), _ = cv2.phaseCorrelate(gr(S[1]), gr(O[1]), hw); SH[k] = (dx / sc, dy / sc)
    buf.pop(0)
rep = {"frames": n, "fps": fps}
for i, nm in enumerate(names):
    s = out[:, i]; v = np.isfinite(s)
    if v.sum() < 30: continue
    med = float(np.nanmedian(s)); p95 = float(np.nanpercentile(s, 95)); order = np.argsort(-np.nan_to_num(s, nan=-1)); peaks = []
    for k in order:
        if len(peaks) >= a.top or not np.isfinite(s[k]): break
        if all(abs(k - q) > int(0.6 * fps) for q in peaks): peaks.append(int(k))
    ph = {L: [round(float(np.nanmean(s[c::L])), 2) for c in range(L)] for L in PER}
    w5 = [round(float(np.nanmean(s[int(t * fps):int((t + 5) * fps)])), 2) for t in range(0, int(n / fps), 5)]
    rep[nm] = {"median": round(med, 2), "p95": round(p95, 2), "amp_median": round(float(np.nanmedian(amp[:, i])), 2), "peaks_s": [[round((a.k0 + k) / fps, 2), round(float(s[k]), 1)] for k in sorted(peaks)],
               "phase_max_over_median": {L: round(max(ph[L]) / max(float(np.median(ph[L])), 1e-6), 2) for L in PER}, "phase": ph, "mean_per_5s": w5}
    print("TWITCH %-5s J median %.2f p95 %.2f | D %.2f | phase " % (nm, med, p95, rep[nm]["amp_median"]) + " ".join("%d:x%.2f" % (L, rep[nm]["phase_max_over_median"][L]) for L in PER) + " | peaks " + " ".join("%.2f(%.0f)" % (t, v_) for t, v_ in rep[nm]["peaks_s"]))
    print("       per 5 s: " + " ".join("%.1f" % x for x in w5))
d2s = lambda x: np.abs(x[2:] + x[:-2] - 2 * x[1:-1])
for i, nm in enumerate(names):
    if np.isfinite(Mo[:, i]).sum() > 30:
        fl_ = d2s(Mo[:, i]); rep[nm + "_x"] = {"pump_median": round(float(np.nanmedian(fl_)), 3), "pump_p95": round(float(np.nanpercentile(fl_, 95)), 3), "J_src": round(float(np.nanmedian(Js[:, i])), 2), "J_res": round(float(np.nanmedian(Jr[:, i])), 2)}
        print("TWITCH %-5s brightness pump (2nd diff of the mean offset) median %.3f p95 %.3f | own jerk: result %.2f source %.2f" % (nm, rep[nm + "_x"]["pump_median"], rep[nm + "_x"]["pump_p95"], rep[nm + "_x"]["J_res"], rep[nm + "_x"]["J_src"]))
if np.isfinite(SH[:, 0]).sum() > 30:
    j2 = np.hypot(d2s(SH[:, 0]), d2s(SH[:, 1])); mag = np.hypot(SH[:, 0], SH[:, 1]); rep["shift"] = {"median_px": round(float(np.nanmedian(mag)), 3), "jitter_median_px": round(float(np.nanmedian(j2)), 3), "jitter_p95_px": round(float(np.nanpercentile(j2, 95)), 3)}
    pk = np.argsort(-np.nan_to_num(j2, nan=-1))[:10]; print("TWITCH patch shift vs the head: |shift| median %.2f px, jitter (2nd diff) median %.3f p95 %.3f px | phase " % (rep["shift"]["median_px"], rep["shift"]["jitter_median_px"], rep["shift"]["jitter_p95_px"]) + " ".join("%d:x%.2f" % (L, max(np.nanmean(j2[c::L]) for c in range(L)) / max(np.nanmedian([np.nanmean(j2[c::L]) for c in range(L)]), 1e-6)) for L in PER) + " | worst s " + " ".join("%.2f" % ((a.k0 + k + 1) / fps) for k in sorted(pk)))
if a.json: json.dump(rep, open(a.json, "w"))
if a.npz: np.savez_compressed(a.npz, j=out, amp=amp, mo=Mo, js=Js, jr=Jr, sh=SH)
