#!/usr/bin/env python3
"""r8 F: temporal TONE stabiliser of the LatentSync layer (user, final_r7 03: "a little fire in his mouth" + the lower face flickers).

Measured (r8_mouth_flicker.py, 03): the inner-mouth frame-to-frame tone flicker comes from the GENERATOR, not from the composite / ESRGAN:
source 25 fps master inner_hf 1.36 (corr with the opening 0.28) -> LatentSync raw layer 2.47 (corr -0.01) -> composite 2.59.
LatentSync generates every frame's mouth interior (dark cavity / teeth / tongue) with its own brightness; the tone jumps between frames
while the opening is smooth. Fix, on the layer (25 fps, its own MediaPipe landmarks), geometry untouched:
  inner mouth   Yin[m] = mean luma in the inner-lip polygon; f = least squares Yin ~ open + area over the video (the part of the tone the
                opening explains); r = Yin - f; target T = f + smooth(r) (median-3, then Gaussian --sigma frames, inside runs of valid frames);
                gain = clip(T / Yin, 1 -/+ --inner-max), ramped in by the opening, applied multiplicatively on a feathered inner-lip mask
                (texture / teeth kept, only the frame's overall interior tone moves toward its neighbours')
  lower face    (--lf on) 3 x 3 cells in face coordinates between the nose base and the chin (lips hull + 0.15 IOD excluded from the
                estimate), cell luma smoothed the same way; the per-cell difference (clip +-(--lf-max)) is interpolated bilinearly over the
                lower face (inner mouth excluded) and ADDED: a low-frequency deflicker that cannot move the jaw / lips
    python scripts/r8/r8_layer_deflicker.py --ls LAYER.mp4 --lm LAYER.mediapipe.npz --out OUT.mp4 [--sigma 1.5 --inner-max 0.3 --lf on]
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import numpy as np, cv2
cv2.setNumThreads(4)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from r8_mouth_flicker import LIPS_INNER, LIPS_OUTER, FACE_OVAL, lower_face_geom


def smooth_runs(x: np.ndarray, valid: np.ndarray, sigma: float, rad: int = 4) -> np.ndarray:
    """median-3 then Gaussian over each run of consecutive valid samples (never across an invalid frame / cut)."""
    out = np.full(len(x), np.nan); i = 0; n = len(x)
    k = np.exp(-0.5 * (np.arange(-rad, rad + 1) / max(sigma, 1e-3)) ** 2)
    while i < n:
        if not valid[i]: i += 1; continue
        j = i
        while j < n and valid[j]: j += 1
        seg = x[i:j].copy()
        if len(seg) >= 3: seg = np.concatenate([[seg[0]], np.median(np.stack([seg[:-2], seg[1:-1], seg[2:]]), 0), [seg[-1]]])
        for t in range(len(seg)):
            lo, hi = max(0, t - rad), min(len(seg), t + rad + 1); w = k[lo - t + rad:hi - t + rad]
            out[i + t] = float((seg[lo:hi] * w).sum() / w.sum())
        i = j
    return out


def face_roi(shape, p, iod):   # face-oval box + 0.3 IOD (all per-frame work happens on this crop)
    q = p[FACE_OVAL, :2]; m = 0.3 * iod; H, W = shape[:2]
    x0, y0 = int(max(0, q[:, 0].min() - m)), int(max(0, q[:, 1].min() - m)); x1, y1 = int(min(W, q[:, 0].max() + m + 1)), int(min(H, q[:, 1].max() + m + 1))
    pc = p.copy(); pc[:, 0] -= x0; pc[:, 1] -= y0
    return (y0, y1, x0, x1), pc


def frame_stats(f, p, a):
    """-> (yin, open, area, cells[9]) of one layer frame (NaN where undefined)"""
    cells = np.full(9, np.nan)
    if not np.isfinite(p[LIPS_INNER + LIPS_OUTER + FACE_OVAL + [33, 263, 2, 152]]).all(): return np.nan, np.nan, np.nan, cells
    iod = float(np.linalg.norm(p[33, :2] - p[263, :2]))
    if iod <= 5: return np.nan, np.nan, np.nan, cells
    (y0, y1, x0, x1), p = face_roi(f.shape, p, iod); f = f[y0:y1, x0:x1]
    Y = (f[..., 2] * 0.299 + f[..., 1] * 0.587 + f[..., 0] * 0.114).astype(np.float32)
    op = float(np.linalg.norm(p[13, :2] - p[14, :2]) / iod); q = p[LIPS_INNER, :2]; ar = float(cv2.contourArea(q.astype(np.float32)) / iod ** 2)
    m = np.zeros(Y.shape, np.uint8); cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1); m = cv2.erode(m, np.ones((3, 3), np.uint8))
    yin = float(Y[m > 0].mean()) if m.sum() >= a.min_px else np.nan
    g = lower_face_geom(Y.shape, p, iod)
    if g is not None:
        ys, xs, ci, cj, est = g; vals = Y[ys, xs]; idx = ci * 3 + cj
        for c in range(9):
            mm = est & (idx == c)
            if mm.sum() >= 15: cells[c] = float(vals[mm].mean())
    return yin, op, ar, cells


def main():
    import av
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ls", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, default=1.5); ap.add_argument("--inner-max", type=float, default=0.3); ap.add_argument("--open-min", type=float, default=0.03)
    ap.add_argument("--open-ramp", type=float, default=0.03); ap.add_argument("--min-px", type=int, default=6); ap.add_argument("--feather", type=float, default=0.04, help="inner mask dilation + blur, IOD")
    ap.add_argument("--lf", choices=["off", "on"], default="on"); ap.add_argument("--lf-max", type=float, default=4.0); ap.add_argument("--crf", type=int, default=10)
    a = ap.parse_args(); lm = np.load(a.lm)["lm"]; t0 = time.perf_counter()
    c = av.open(a.ls); s = c.streams.video[0]; s.thread_type = "AUTO"; rate = s.average_rate; n = len(lm)
    yin = np.full(n, np.nan); op = np.full(n, np.nan); ar = np.full(n, np.nan); cells = np.full((n, 9), np.nan); k = 0
    for fr in c.decode(video=0):
        if k >= n: break
        yin[k], op[k], ar[k], cells[k] = frame_stats(fr.to_ndarray(format="bgr24"), lm[k], a); k += 1
    c.close(); n = k
    # inner-mouth target
    v = np.isfinite(yin[:n]) & np.isfinite(op[:n]) & (op[:n] >= a.open_min); gain = np.ones(n)
    if v.sum() > 30:
        X = np.stack([np.ones(v.sum()), op[:n][v], ar[:n][v]], 1); coef, *_ = np.linalg.lstsq(X, yin[:n][v], rcond=None)
        f = np.full(n, np.nan); f[v] = X @ coef; r = yin[:n] - f; rs = smooth_runs(r, v, a.sigma)
        tgt = f + rs; wo = np.clip((op[:n] - a.open_min) / a.open_ramp, 0, 1)
        g = np.where(v, np.clip(tgt / np.maximum(yin[:n], 1.0), 1 - a.inner_max, 1 + a.inner_max), 1.0); gain = 1 + wo * (g - 1); gain[~v] = 1.0
    # lower-face targets
    dcell = np.zeros((n, 9))
    if a.lf == "on":
        for cc in range(9):
            vc = np.isfinite(cells[:n, cc]); sm = smooth_runs(cells[:n, cc], vc, a.sigma); dcell[:, cc] = np.where(vc, np.clip(sm - cells[:n, cc], -a.lf_max, a.lf_max), 0)
    t1 = time.perf_counter()
    c = av.open(a.ls); s = c.streams.video[0]; s.thread_type = "AUTO"; o = av.open(a.out, "w")
    st = o.add_stream("libx264", rate=rate); st.width, st.height = s.codec_context.width, s.codec_context.height; st.pix_fmt = "yuv420p"; st.options = {"crf": str(a.crf), "preset": "medium"}; k = 0; nin = nlf = 0
    for fr in c.decode(video=0):
        if k >= n: break
        img = fr.to_ndarray(format="bgr24"); p = lm[k]
        if np.isfinite(p[LIPS_INNER + FACE_OVAL + [33, 263, 2, 152]]).all() and (abs(gain[k] - 1) > 1e-3 or np.abs(dcell[k]).max() > 1e-3):
            iod = float(np.linalg.norm(p[33, :2] - p[263, :2])); (y0, y1, x0, x1), p = face_roi(img.shape, p, iod); full_ = img; img = img[y0:y1, x0:x1]; x = img.astype(np.float32)
            mi = np.zeros(img.shape[:2], np.uint8); cv2.fillPoly(mi, [np.round(p[LIPS_INNER, :2]).astype(np.int32)], 1)
            fp = max(1.0, a.feather * iod); mi = cv2.GaussianBlur(cv2.dilate(mi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(fp) + 1,) * 2)).astype(np.float32), (0, 0), fp / 2)
            if abs(gain[k] - 1) > 1e-3: x *= (1 + (gain[k] - 1) * mi)[..., None]; nin += 1
            if np.abs(dcell[k]).max() > 1e-3:
                g = lower_face_geom(img.shape[:2], p, iod, full=True)
                if g is not None:
                    ys, xs, fu, fv, wreg = g   # continuous cell coordinates and the region weight (feathered, inner mouth excluded below)
                    grid = dcell[k].reshape(3, 3).astype(np.float32)
                    d = cv2.remap(grid, fu.astype(np.float32)[None, :], fv.astype(np.float32)[None, :], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
                    field = np.zeros(img.shape[:2], np.float32); field[ys, xs] = d * wreg
                    field = cv2.GaussianBlur(field, (0, 0), max(1.0, 0.08 * iod)) * (1 - mi)
                    x += field[..., None]; nlf += 1
            full_ = full_.copy(); full_[y0:y1, x0:x1] = np.clip(x, 0, 255).round().astype(np.uint8); img = full_
        for pk in st.encode(av.VideoFrame.from_ndarray(img, format="bgr24")): o.mux(pk)
        k += 1
    for pk in st.encode(): o.mux(pk)
    o.close(); c.close()
    rep = {"frames": n, "inner_frames_changed": nin, "lf_frames_changed": nlf, "gain_abs_mean": float(np.mean(np.abs(gain - 1))), "gain_abs_p95": float(np.percentile(np.abs(gain - 1), 95)),
           "lf_abs_mean": float(np.mean(np.abs(dcell))), "seconds": {"stats": round(t1 - t0, 1), "apply": round(time.perf_counter() - t1, 1)}, "args": vars(a)}
    pathlib.Path(a.out).with_suffix(".dfk.json").write_text(json.dumps(rep, indent=1)); print("== deflicker", json.dumps({k_: v_ for k_, v_ in rep.items() if k_ != "args"}), flush=True)


if __name__ == "__main__":
    main()
