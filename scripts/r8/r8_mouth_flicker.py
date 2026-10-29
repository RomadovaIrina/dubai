#!/usr/bin/env python3
"""r8 F: mouth-interior / lower-face FLICKER metric (user, final_r7 03: "the inside of the mouth goes light <-> dark from frame to frame,
uncorrelated with the opening, like a little fire in his mouth; the lower face flickers too").

Per video (each with its OWN MediaPipe landmarks, cached next to it as <stem>.flk_mp.npz unless --lm is given):
  open[t]   inner opening 13-14 / IOD;  area[t] = inner-lip polygon area / IOD^2
  Yin[t]    mean luma inside the inner-lip polygon (eroded 1 px), on frames with open >= --open-min (an interior exists)
  model     Yin ~ a + b*open + c*area (least squares on the video itself): the part of the interior tone that the opening explains
  r[t]      Yin - model = the tone NOT explained by the opening
  metrics   inner_dY      median / p90 |Yin[t] - Yin[t-1]| over consecutive open frames
            inner_hf      median / p90 |r[t] - (r[t-1] + r[t+1]) / 2|  (frame-to-frame flicker of the unexplained tone; slow drift excluded)
            inner_events  frames with inner_hf > --event-thr luma, per second of open-mouth video
            corr_open     Pearson(Yin, open)  (source: the interior darkens / brightens with the opening)
  lower face: the region nose base .. chin inside the face oval, minus the outer-lip hull dilated by 0.15 IOD, split into a 3 x 3 grid
            in face coordinates; per cell mean luma; lf_hf = median over frames of the mean |cell[t] - (cell[t-1] + cell[t+1]) / 2|
  retime    for a 25 fps layer shown at the source fps: corr of the per-frame inner_hf / lf_hf with the retime phase distance to the nearest
            25 fps frame (|frac(k*25/fps) - round|) -> high = the 25 -> 30 fps interpolation drives the flicker
    python scripts/r8/r8_mouth_flicker.py --video src:SRC.mp4 m11:OUT.mp4 [layer:LS_25fps.mp4:LM.npz] --out J.json [--csv-dir D]
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def lower_face_geom(shape, p, iod, full=False):
    """The lower face (nose base .. chin, inside the face oval) in face coordinates (u = eye line, v = down), split into 3 x 3 cells.
    full=False -> (ys, xs, ci, cj, est): pixels, cell row / column, est = outside the outer-lip hull + 0.15 IOD (the tone estimate)
    full=True  -> (ys, xs, fu, fv, wreg): pixels, continuous cell coordinates (cell centres at 0, 1, 2), feathered region weight"""
    u = (p[263, :2] - p[33, :2]) / iod; v = np.array([-u[1], u[0]]); v = v if v[1] > 0 else -v
    ov = np.zeros(shape[:2], np.uint8); cv2.fillPoly(ov, [np.round(p[FACE_OVAL, :2]).astype(np.int32)], 1)
    x0 = p[2, :2]; ych = float((p[152, :2] - p[2, :2]) @ v)
    if ych <= 5: return None
    ys, xs = np.nonzero(ov)
    d = np.stack([xs - x0[0], ys - x0[1]], 1).astype(np.float32); sv = d @ v; su = d @ u
    sel = (sv > 0) & (sv < ych)
    if sel.sum() <= 50: return None
    uu = (p[FACE_OVAL, :2] - x0) @ u; umin, umax = float(uu.min()), float(uu.max())
    ys, xs, sv, su = ys[sel], xs[sel], sv[sel], su[sel]
    fu = (su - umin) / (umax - umin + 1e-6) * 3; fv = sv / ych * 3
    if not full:
        lip = np.zeros(shape[:2], np.uint8); cv2.fillPoly(lip, [np.round(cv2.convexHull(p[LIPS_OUTER, :2].astype(np.float32))[:, 0]).astype(np.int32)], 1)
        r_ = max(1, int(round(0.15 * iod))); lip = cv2.dilate(lip, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1, 2 * r_ + 1)))
        return ys, xs, np.clip(fv.astype(int), 0, 2), np.clip(fu.astype(int), 0, 2), lip[ys, xs] == 0
    reg = np.zeros(shape[:2], np.uint8); reg[ys, xs] = 1
    wreg = np.clip(cv2.distanceTransform(reg, cv2.DIST_L2, 3)[ys, xs] / max(1.0, 0.12 * iod), 0, 1)
    return ys, xs, np.clip(fu - 0.5, 0, 2), np.clip(fv - 0.5, 0, 2), wreg


def landmarks(video: pathlib.Path, lm_path: str | None):
    if lm_path: return np.load(lm_path)["lm"]
    import clean_face_stack as cfs
    out = video.with_suffix(".flk_mp.npz")
    return np.load(cfs.run_mediapipe(video, out))["lm"]


def hf(x):   # |x[t] - (x[t-1] + x[t+1]) / 2|, NaN where a neighbour is missing
    o = np.full(len(x), np.nan); o[1:-1] = np.abs(x[1:-1] - 0.5 * (x[:-2] + x[2:])); return o


def measure(video: pathlib.Path, lm: np.ndarray, a) -> tuple[dict, dict]:
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; n = len(lm)
    yin = np.full(n, np.nan); op = np.full(n, np.nan); ar = np.full(n, np.nan); cells = np.full((n, 9), np.nan); npx = np.zeros(n)
    k = 0
    while k < n:
        ok, f = cap.read()
        if not ok: break
        p = lm[k]
        if np.isfinite(p[LIPS_INNER + LIPS_OUTER + FACE_OVAL + [33, 263, 2, 152]]).all():
            iod = float(np.linalg.norm(p[33, :2] - p[263, :2]))
            if iod > 5:
                Y = (f[..., 2] * 0.299 + f[..., 1] * 0.587 + f[..., 0] * 0.114).astype(np.float32)
                op[k] = np.linalg.norm(p[13, :2] - p[14, :2]) / iod; q = p[LIPS_INNER, :2]; ar[k] = cv2.contourArea(q.astype(np.float32)) / iod ** 2
                m = np.zeros(Y.shape, np.uint8); cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
                m = cv2.erode(m, np.ones((3, 3), np.uint8)); npx[k] = m.sum()
                if m.sum() >= a.min_px: yin[k] = float(Y[m > 0].mean())
                g = lower_face_geom(Y.shape, p, iod)
                if g is not None:
                    ys, xs, ci, cj, est = g; vals = Y[ys, xs]; idx = ci * 3 + cj
                    for c in range(9):
                        mm = est & (idx == c)
                        if mm.sum() >= 15: cells[k, c] = float(vals[mm].mean())
        k += 1
    cap.release(); n = k; yin, op, ar, cells, npx = yin[:n], op[:n], ar[:n], cells[:n], npx[:n]
    good = np.isfinite(yin) & np.isfinite(op) & (op >= a.open_min)
    R = {"fps": round(fps, 3), "frames": n, "open_frames": int(good.sum()), "inner_px_median": float(np.median(npx[good])) if good.any() else None}
    per = {"yin": yin, "open": op, "area": ar}
    if good.sum() > 30:
        X = np.stack([np.ones(good.sum()), op[good], ar[good]], 1); coef, *_ = np.linalg.lstsq(X, yin[good], rcond=None)
        r = np.full(n, np.nan); r[good] = yin[good] - X @ coef
        dY = np.abs(np.diff(np.where(good, yin, np.nan))); h = hf(r)
        R.update({"corr_open": float(np.corrcoef(yin[good], op[good])[0, 1]), "explained_frac": float(1 - np.var(r[good]) / max(np.var(yin[good]), 1e-6)),
                  "inner_dY_med": float(np.nanmedian(dY)), "inner_dY_p90": float(np.nanpercentile(dY, 90)),
                  "inner_hf_med": float(np.nanmedian(h)), "inner_hf_p90": float(np.nanpercentile(h, 90)),
                  "inner_events_per_s": float(np.nansum(h > a.event_thr) / (good.sum() / fps)), "yin_med": float(np.median(yin[good]))})
        per["inner_hf"] = h
        top = np.argsort(-np.nan_to_num(h, nan=-1))[:12]; R["inner_top_events_s"] = [[round(int(t) / fps, 2), round(float(h[t]), 1)] for t in sorted(top)]
    lfh = np.nanmean(np.stack([hf(cells[:, c]) for c in range(9)], 1), 1) if np.isfinite(cells).any() else np.full(n, np.nan)
    per["lf_hf"] = lfh
    if np.isfinite(lfh).sum() > 30:
        cell_hf = [float(np.nanmedian(hf(cells[:, c]))) for c in range(9)]
        R.update({"lf_hf_med": float(np.nanmedian(lfh)), "lf_hf_p90": float(np.nanpercentile(lfh, 90)), "lf_cells_hf_med": [round(x, 3) for x in cell_hf]})
    if abs(fps - 25) > 0.5:   # retime phase of a 25 fps layer shown at fps
        ph = (np.arange(n) * 25.0 / fps) % 1.0; dist = np.minimum(ph, 1 - ph)
        for key in ("inner_hf", "lf_hf"):
            x = per.get(key)
            if x is not None and np.isfinite(x).sum() > 30:
                ok = np.isfinite(x); R[f"{key}_corr_retime_phase"] = float(np.corrcoef(x[ok], dist[ok])[0, 1])
                R[f"{key}_by_phase"] = {f"{b:.2f}": float(np.nanmedian(np.where(np.isclose(dist, b, atol=0.02), x, np.nan))) for b in sorted(set(np.round(dist, 2)))}
    return R, per


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", nargs="+", required=True, help="label:video[:landmarks.npz]")
    ap.add_argument("--open-min", type=float, default=0.05); ap.add_argument("--min-px", type=int, default=6); ap.add_argument("--event-thr", type=float, default=8.0)
    ap.add_argument("--out", required=True); ap.add_argument("--csv-dir")
    a = ap.parse_args(); res = {}
    for spec in a.video:
        parts = spec.split(":"); lab, vid = parts[0], pathlib.Path(parts[1]); lmp = parts[2] if len(parts) > 2 else None
        lm = landmarks(vid, lmp); R, per = measure(vid, lm, a); res[lab] = R
        print(f"== {lab}: " + ", ".join(f"{k} {v:.3g}" for k, v in R.items() if isinstance(v, float)), flush=True)
        if a.csv_dir:
            d = pathlib.Path(a.csv_dir); d.mkdir(parents=True, exist_ok=True); keys = list(per)
            np.savetxt(d / f"{lab}.csv", np.stack([per[k] for k in keys], 1), delimiter=",", header=",".join(keys), fmt="%.4f")
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.out).write_text(json.dumps(res, indent=1)); print("== written", a.out)


if __name__ == "__main__":
    main()
