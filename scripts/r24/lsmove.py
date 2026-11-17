#!/usr/bin/env python3
"""r24 (coordinator 17.11: what breaks LatentSync on a MOVING face). Compares a final with its source frame by frame (same fps, same frame
count; landmarks of both from mp_landmarks.py) and splits the patch defects by cause:
  jitter  : residual of the drawn lower face vs the source (final lm - source lm, points inside the patch but off the lips: nose base, chin,
            jaw, cheeks) after a +-2 frame Gaussian high-pass, in 1/1000 of the face height; the same for points OUTSIDE the patch (eyes,
            brow, forehead) = the detector's noise floor, reported next to it
  slide   : the low-pass part of the same residual (the patch drifting / lagging behind the head)
  flicker : mean |frame-to-frame change of (final - source)| luma in the lower-face box (the drawn patch only)
  seams   : jitter and flicker folded by the generator's 25 fps frame phase mod 16 and mod 8 (max phase / median phase)
  causes  : the means in quartiles of head speed, |yaw|, source blur (Laplacian of the face at a fixed size) and jaw speed
  worst   : the 1 s windows with the highest jitter / flicker (for the clips)
    lsmove.py --src src.mp4 --fin final.mp4 --lm-src s.npz --lm-fin f.npz --out x.json [--fps 30]"""
import argparse, json
import numpy as np, cv2
ap = argparse.ArgumentParser()
for k in ("src", "fin", "lm-src", "lm-fin", "out"): ap.add_argument("--" + k, required=True)
ap.add_argument("--voice", default=""); ap.add_argument("--fps", type=float, default=0.0); ap.add_argument("--gen-fps", type=float, default=25.0)
a = ap.parse_args()
S = np.load(a.lm_src); F = np.load(a.lm_fin)
ls, lf = S["lm"][..., :2].astype(np.float64), F["lm"][..., :2].astype(np.float64)
n = min(len(ls), len(lf)); ls, lf = ls[:n], lf[:n]
cap = cv2.VideoCapture(a.src); fps = a.fps or cap.get(cv2.CAP_PROP_FPS) or 25.0; cap.release()
IN = [2, 98, 327, 94, 152, 148, 377, 176, 400, 172, 397, 136, 365, 205, 425, 216, 436, 18, 200]   # inside the patch, off the lips
OUT = [33, 263, 133, 362, 168, 6, 9, 105, 334, 10]                                                  # outside the patch
ok = np.isfinite(ls).all((1, 2)) & np.isfinite(lf).all((1, 2))
fh = np.where(ok, ls[:, :, 1].max(1) - ls[:, :, 1].min(1), np.nan)
def gs(x, s):
    r = int(3 * s) + 1; g = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2); g /= g.sum()
    y = x.copy(); m = np.isfinite(y)
    if m.sum() < 3: return y
    idx = np.arange(len(y)); y = np.interp(idx, idx[m], y[m])
    return np.convolve(np.pad(y, r, mode="edge"), g, mode="valid")
def resid(P):
    d = (lf[:, P] - ls[:, P]) / fh[:, None, None]   # (n, k, 2) in face heights
    hp = np.zeros((n, len(P))); lp = np.zeros((n, len(P)))
    for j in range(len(P)):
        for c in (0, 1):
            x = d[:, j, c]; s = gs(x, 2.0); hp[:, j] += (x - s) ** 2; lp[:, j] += gs(x, 6.0) ** 2
    return 1000 * np.sqrt(hp).mean(1), 1000 * np.sqrt(lp).mean(1)
J_in, SL_in = resid(IN); J_out, SL_out = resid(OUT)
J_in[~ok] = np.nan; J_out[~ok] = np.nan; SL_in[~ok] = np.nan
c = ls.mean(1); v = np.r_[np.nan, np.linalg.norm(np.diff(c, axis=0), axis=1)] / fh * 1000   # head speed, 1/1000 face height per frame
dl, dr = np.abs(ls[:, 1, 0] - ls[:, 234, 0]), np.abs(ls[:, 454, 0] - ls[:, 1, 0]); yaw = np.abs(dl - dr) / np.maximum(dl + dr, 1e-3)
jaw = np.r_[np.nan, np.abs(np.diff(ls[:, 152, 1] - ls[:, 1, 1]))] / fh * 1000                       # jaw opening speed
# flicker of (final - source) in the lower-face box + blur of the source face
c1, c2 = cv2.VideoCapture(a.src), cv2.VideoCapture(a.fin); prev = None; FL = np.full(n, np.nan); LAP = np.full(n, np.nan); FLOW = np.full(n, np.nan); pg = None; DM = np.full(n, np.nan)   # DM: |final - source| in the mouth box (drawn or kept source); FLOW: residual optical flow (final - source) in the patch off the lips, 1/1000 face height
for k in range(n):
    o1, f1 = c1.read(); o2, f2 = c2.read()
    if not (o1 and o2): break
    if f1.shape != f2.shape: f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]), interpolation=cv2.INTER_AREA)
    ds = 360.0 / max(f1.shape[:2]); q1 = cv2.cvtColor(cv2.resize(f1, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY); q2 = cv2.cvtColor(cv2.resize(f2, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    if pg is not None and ok[k]:
        fa = cv2.calcOpticalFlowFarneback(pg[0], q1, None, 0.5, 3, 15, 3, 5, 1.2, 0); fb = cv2.calcOpticalFlowFarneback(pg[1], q2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        pp = ls[k] * ds; m = np.zeros(q1.shape, np.uint8); cv2.fillConvexPoly(m, cv2.convexHull(pp[[2, 205, 425, 152, 172, 397, 136, 365, 148, 377]].astype(np.int32)), 1)
        cv2.fillConvexPoly(m, cv2.convexHull(pp[[61, 291, 0, 17, 37, 267, 84, 314]].astype(np.int32)), 0); m = cv2.erode(m, np.ones((3, 3), np.uint8))
        if m.sum() > 20: FLOW[k] = 1000 * float(np.linalg.norm((fb - fa)[m > 0], axis=1).mean()) / (fh[k] * ds)
    pg = (q1, q2)
    if ok[k]:
        qq = ls[k]; mx0, mx1 = int(max(0, qq[61, 0] - 10)), int(qq[291, 0] + 10); my0, my1 = int(max(0, qq[0, 1] - 15)), int(qq[17, 1] + 15)
        if mx1 - mx0 > 4 and my1 - my0 > 4: DM[k] = float(np.abs(cv2.cvtColor(f1[my0:my1, mx0:mx1], cv2.COLOR_BGR2GRAY).astype(np.float32) - cv2.cvtColor(f2[my0:my1, mx0:mx1], cv2.COLOR_BGR2GRAY).astype(np.float32)).mean())
    if not ok[k]: prev = None; continue
    p = ls[k]; x0, x1 = p[[234, 454], 0].min(), p[[234, 454], 0].max(); y0, y1 = p[2, 1], p[152, 1] + 0.05 * fh[k]
    x0, y0 = int(max(0, x0)), int(max(0, y0)); x1, y1 = int(min(f1.shape[1], x1)), int(min(f1.shape[0], y1))
    if x1 - x0 < 8 or y1 - y0 < 8: prev = None; continue
    g1 = cv2.cvtColor(f1[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32); g2 = cv2.cvtColor(f2[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    sc = 128.0 / max(g1.shape); d = cv2.resize(g2 - g1, (128, int(round(128 * g1.shape[0] / g1.shape[1]))) if g1.shape[1] >= g1.shape[0] else (int(round(128 * g1.shape[1] / g1.shape[0])), 128), interpolation=cv2.INTER_AREA)
    LAP[k] = cv2.Laplacian(cv2.resize(g1, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA), cv2.CV_32F).var()
    if prev is not None and prev.shape == d.shape: FL[k] = float(np.abs(d - prev).mean())
    prev = d
blur = -LAP / np.nanmedian(LAP)   # higher = blurrier
# the drawn mouth itself: opening (13-14) and width (61-291) in 1/1000 face height; twitch = |x - smooth(x, 1.5 frames)|, the source's own as reference
def lipser(L): return 1000 * np.abs(L[:, 14, 1] - L[:, 13, 1]) / fh, 1000 * np.abs(L[:, 291, 0] - L[:, 61, 0]) / fh
def tw(x): return np.abs(x - gs(x, 1.5))
oF, wF = lipser(lf); oS, wS = lipser(ls); TWF = tw(oF) + tw(wF); TWS = tw(oS) + tw(wS)
STK = {}
if a.voice:   # sticking: the dub voice is on (frame RMS > 15 % of its p95) but the drawn mouth is nearly shut (< 25 % of its own p90 opening)
    from scipy.io import wavfile
    sr, wv = wavfile.read(a.voice); wv = wv.astype(np.float64); wv = wv.mean(1) if wv.ndim > 1 else wv
    hop = sr / fps; rms = np.array([np.sqrt(np.mean(wv[int(i * hop):int((i + 1) * hop)] ** 2)) if int(i * hop) < len(wv) else 0.0 for i in range(n)])
    on = rms > 0.15 * np.percentile(rms, 95); shut = oF < 0.25 * np.nanpercentile(oF, 90); st_ = on & shut
    d_ = np.diff(np.r_[0, st_.astype(int), 0]); runs = [(s_, e_) for s_, e_ in zip(np.flatnonzero(d_ == 1), np.flatnonzero(d_ == -1)) if e_ - s_ >= 3]
    mm = np.isfinite(oF)
    STK = {"voice_on_share": round(float(on.mean()), 3), "stuck_share_of_voice": round(float(st_.sum() / max(on.sum(), 1)), 3), "stuck_runs_ge3": len(runs),
           "stuck_runs_s": [[round(s_ / fps, 2), round(e_ / fps, 2)] for s_, e_ in runs][:12], "corr_opening_voice": round(float(np.corrcoef(oF[mm], rms[mm])[0, 1]), 3),
           "opening_when_voice_p50": round(float(np.nanmedian(oF[on])), 1), "opening_when_silent_p50": round(float(np.nanmedian(oF[~on])), 1)}
drawn = DM > 4.0   # (encode noise ~2) frames where the generator's mouth is on screen (elsewhere the source is kept)
drawn = np.convolve(drawn.astype(float), np.ones(5) / 5, mode="same") > 0.2
for X_ in (J_in, J_out, SL_in, SL_out, FL, FLOW, TWF, TWS): X_[~drawn] = np.nan
t = np.arange(n) / fps; ph = np.floor(t * a.gen_fps + 1e-6).astype(int)
def fold(x, m):
    r = [float(np.nanmean(x[(ph % m) == i])) for i in range(m)]; md = float(np.nanmedian(r)); return {"by_phase": [round(q, 2) for q in r], "max_over_median": round(max(r) / max(md, 1e-6), 3), "argmax": int(np.argmax(r))}
def quart(x, y):
    m = np.isfinite(x) & np.isfinite(y); qs = np.nanpercentile(x[m], [25, 50, 75]); b = np.digitize(x[m], qs)
    return [round(float(np.mean(y[m][b == i])), 2) if (b == i).any() else None for i in range(4)]
def worst(x, k=5):
    w = int(round(fps)); s = np.array([np.nanmean(x[i:i + w]) for i in range(0, max(1, n - w), max(1, w // 2))]); o = np.argsort(-np.nan_to_num(s, nan=-1))[:k]
    return [[round(float(i * (w // 2) / fps), 1), round(float(s[i]), 2)] for i in o]
nn = lambda x: round(float(np.nanmean(x)), 2)
R = {"drawn_share": round(float(drawn.mean()), 3), "frames": n, "fps": fps, "face_found": int(ok.sum()),
     "jitter_in_patch": nn(J_in), "jitter_outside(noise_floor)": nn(J_out), "jitter_in_p95": round(float(np.nanpercentile(J_in, 95)), 2),
     "lip_twitch_final": nn(TWF), "lip_twitch_source": nn(TWS), "lip_twitch_final_p95": round(float(np.nanpercentile(TWF, 95)), 2),
     "opening_final_p50_p90": [round(float(np.nanpercentile(oF, q)), 1) for q in (50, 90)], "opening_source_p50_p90": [round(float(np.nanpercentile(oS, q)), 1) for q in (50, 90)],
     "slide_in_patch": nn(SL_in), "slide_outside": nn(SL_out), "flow_resid": nn(FLOW), "flow_resid_p95": round(float(np.nanpercentile(FLOW, 95)), 2), "flicker": nn(FL), "flicker_p95": round(float(np.nanpercentile(FL, 95)), 2),
     "seams": {"jitter_mod16": fold(J_in - J_out, 16), "jitter_mod8": fold(J_in - J_out, 8), "flicker_mod16": fold(FL, 16), "flow_mod16": fold(FLOW, 16), "flow_mod8": fold(FLOW, 8), "lip_twitch_mod16": fold(TWF, 16), "lip_twitch_mod8": fold(TWF, 8), "flicker_mod8": fold(FL, 8)},
     "by_quartile(low->high)": {k: {"jitter": quart(x, J_in - J_out), "flow": quart(x, FLOW), "flicker": quart(x, FL), "slide": quart(x, SL_in)} for k, x in
                                 (("head_speed", v), ("yaw", yaw), ("blur", blur), ("jaw_speed", jaw))},
     "quartile_edges": {"head_speed": [round(float(q), 1) for q in np.nanpercentile(v, [25, 50, 75])], "yaw": [round(float(q), 3) for q in np.nanpercentile(yaw, [25, 50, 75])]},
     "sticking": STK, "worst_jitter_1s": worst(J_in - J_out), "worst_flicker_1s": worst(FL), "worst_lip_twitch_1s": worst(TWF), "worst_flow_1s": worst(FLOW)}
np.savez_compressed(a.out.replace(".json", "_ser.npz"), J_in=J_in, J_out=J_out, SL=SL_in, FL=FL, FLOW=FLOW, TWF=TWF, TWS=TWS, oF=oF, oS=oS, v=v, yaw=yaw, blur=blur, jaw=jaw, t=t)
json.dump(R, open(a.out, "w"), indent=1); print("LSMOVE " + json.dumps(R), flush=True)
