#!/usr/bin/env python3
"""Round 3 track B: flicker / edge-crawl / slide measure of the PASTED region, motion-compensated by the SOURCE's own motion.

For each pair of consecutive frames (k-1, k): dense DIS flow of the REFERENCE (source, or the 25 fps master for a 25 fps output) from k to
k-1 warps frame k-1 of BOTH videos onto frame k: r_ref = |ref_k - warp(ref_k-1)|, r_out = |out_k - warp(out_k-1)| (grey levels). Where the
output equals the reference or moves exactly with it, r_out = r_ref; a pasted region that slides against the head, flickers in colour or
texture, or whose mask edge crawls over the skin gives r_out > r_ref. excess = mean(r_out - r_ref) per region:
  band_up   mask border band (alpha 0.1-0.9 of the composite ellipse, recomputed from the reference landmarks) above the mouth-corner line
  band_low  the same band below it (chin / jaw: partly legitimate, the dub jaw articulates differently)
  skin_up   inside the mask (alpha > 0.9), lips dilated by 0.25 mouth width excluded, above the mouth-corner line (nose base, cheeks, upper lip skin)
  chin      inside the mask, below the mouth-corner line, lips excluded (articulation: reported, not ranked)
  upper     eyes / brows (untouched pixels: encoder noise floor)
colour flicker: c_k = mean Lab(out_k - ref_k) over skin_up; flicker_k = |c_k - c_k-1| (a colour / brightness offset that jumps between frames,
e.g. at LatentSync window starts). sharp = mean |Laplacian| of out / ref over skin_up + lips.
Per second: score = excess(band_up) + excess(skin_up) + flicker; top seconds ranked (1 = worst).
    python scripts/quality/r3_flicker.py --video OUT.mp4 --ref SRC.mp4 --lm timing/<id>_source_lm.npz [--ref-master timing.json] --out x.json [--frames A:B]
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pilot"))
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
UPPER = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 33, 133, 362, 263, 168]


def frames(path):
    import av
    c = av.open(path); s = c.streams.video[0]; s.thread_type = "AUTO"
    for fr in c.decode(s): yield fr.to_ndarray(format="bgr24")
    c.close()


def main():
    import cv2, clean_face_stack as cfs
    from fps_composite import ellipse_alpha
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--lm", required=True)
    ap.add_argument("--ref-master", default=None, help="timing json: the reference is the 25 fps master (frame i = source frame j[i]) -> landmarks lm[j[i]]")
    ap.add_argument("--fps", type=float, default=None, help="frames per second for the per-second ranking (default: from the timestamps in --lm, or 25 with --ref-master)")
    ap.add_argument("--frames", default=None); ap.add_argument("--edge-px", type=float, default=12.0); ap.add_argument("--out", required=True); ap.add_argument("--series", action="store_true")
    a = ap.parse_args()
    z = np.load(a.lm); lm = z["lm"].astype(np.float64)
    if a.ref_master:
        j = np.asarray(json.loads(pathlib.Path(a.ref_master).read_text())["j"]); lm = lm[np.clip(j, 0, len(lm) - 1)]; tt = np.arange(len(lm)) / 25.0
    else: tt = z["t"]
    n = len(lm); A, B = (0, n) if not a.frames else [int(x) for x in a.frames.split(":")]
    W = H = None
    dets = cfs.ls_detections(lm.astype(np.float32), 10 ** 5, 10 ** 5, calib=True)
    bb = np.array([d[0] if d[0] is not None else (np.nan,) * 4 for d in dets], np.float64)
    for c in range(4):   # +-3 frame mean, as the composite masks
        col = bb[:, c].copy()
        for i in np.flatnonzero(np.isfinite(col)): bb[i, c] = np.nanmean(col[max(0, i - 3):i + 4])
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    keys = ("band_up", "band_low", "skin_up", "chin", "upper"); ex = {k: np.full(n, np.nan) for k in keys}; rr = {k: np.full(n, np.nan) for k in keys}
    col_off = np.full((n, 3), np.nan); sharp = np.full((n, 2), np.nan)
    prev = None
    for k, (o, r) in enumerate(zip(frames(a.video), frames(a.ref))):
        if k >= B: break
        if k < A - 1: continue
        H, W = r.shape[:2]
        cur = (o, r)
        if prev is None or k < A or not np.isfinite(bb[k]).all() or not np.isfinite(lm[k]).all():
            prev = cur; continue
        L = lm[k]; alpha, roi = ellipse_alpha(bb[k], H, W, a.edge_px)
        if alpha is None: prev = cur; continue
        ya, yb, xa, xb = roi; iod = float(np.linalg.norm(L[33] - L[263]))
        # include the upper face in the working box (noise floor)
        Y0 = int(max(0, min(ya, L[UPPER, 1].min() - 0.3 * iod))); X0 = int(max(0, min(xa, L[UPPER, 0].min() - 0.2 * iod))); Y1, X1 = yb, int(min(W, max(xb, L[UPPER, 0].max() + 0.2 * iod)))
        al = np.zeros((Y1 - Y0, X1 - X0), np.float32); al[ya - Y0:yb - Y0, xa - X0:xb - X0] = alpha
        g = [cv2.cvtColor(x[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY).astype(np.float32) for x in (o, r, prev[0], prev[1])]
        F = dis.calc(g[1].astype(np.uint8), g[3].astype(np.uint8), None)   # ref k -> ref k-1 (backward)
        yy, xx = np.mgrid[0:Y1 - Y0, 0:X1 - X0].astype(np.float32); mx, my = xx + F[..., 0], yy + F[..., 1]
        wo = cv2.remap(g[2], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); wr = cv2.remap(g[3], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ro, rf = np.abs(g[0] - wo), np.abs(g[1] - wr)
        Ls = L[:, :2] - [X0, Y0]; mw = float(np.linalg.norm(L[61] - L[291])); ymc = float((Ls[61, 1] + Ls[291, 1]) / 2)
        lip = np.zeros_like(al, np.uint8); cv2.fillPoly(lip, [np.round(Ls[LIPS_OUTER]).astype(np.int32)], 1)
        lip = cv2.dilate(lip, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.25 * mw) + 1,) * 2)) > 0
        up = yy < ymc
        upr = np.zeros_like(al, np.uint8); cv2.fillPoly(upr, [np.round(cv2.convexHull(Ls[UPPER].astype(np.float32))[:, 0]).astype(np.int32)], 1)
        valid = (mx > 1) & (my > 1) & (mx < X1 - X0 - 2) & (my < Y1 - Y0 - 2)
        R = {"band_up": (al > 0.1) & (al < 0.9) & up, "band_low": (al > 0.1) & (al < 0.9) & ~up, "skin_up": (al > 0.9) & ~lip & up, "chin": (al > 0.9) & ~lip & ~up, "upper": upr > 0}
        for kk, m in R.items():
            m = m & valid
            if m.sum() > 20: ex[kk][k] = float((ro[m] - rf[m]).mean()); rr[kk][k] = float(rf[m].mean())
        m = R["skin_up"]
        if m.sum() > 20:
            lo = cv2.cvtColor(o[Y0:Y1, X0:X1], cv2.COLOR_BGR2LAB).astype(np.float32); lr = cv2.cvtColor(r[Y0:Y1, X0:X1], cv2.COLOR_BGR2LAB).astype(np.float32)
            col_off[k] = (lo[m] - lr[m]).mean(0)
            ms = (m | (lip & (al > 0.5)))
            sharp[k] = [np.abs(cv2.Laplacian(g[0], cv2.CV_32F))[ms].mean(), np.abs(cv2.Laplacian(g[1], cv2.CV_32F))[ms].mean()]
        prev = cur
    fl = np.full(n, np.nan); fl[1:] = np.linalg.norm(np.diff(col_off, axis=0), axis=1)
    fps = a.fps or (25.0 if a.ref_master else (n - 1) / max(tt[-1] - tt[0], 1e-6))
    sec = np.floor(np.asarray(tt[:n]) - tt[0]).astype(int) if not a.ref_master else (np.arange(n) // 25)

    def st(x):
        x = x[np.isfinite(x)]
        return {"mean": round(float(x.mean()), 3), "p95": round(float(np.percentile(x, 95)), 3), "n": int(len(x))} if len(x) else None
    score = np.nan_to_num(ex["band_up"]) + np.nan_to_num(ex["skin_up"]) + np.nan_to_num(fl)
    per = {}
    for s in np.unique(sec[A:B]):
        sel = (sec == s) & np.isfinite(ex["skin_up"])
        if sel.sum() >= 5: per[int(s)] = {"score": round(float(score[sel].mean()), 3), "band_up": round(float(np.nanmean(ex["band_up"][sel])), 3), "skin_up": round(float(np.nanmean(ex["skin_up"][sel])), 3),
                                          "flicker": round(float(np.nanmean(fl[sel])), 3), "band_low": round(float(np.nanmean(ex["band_low"][sel])), 3) if np.isfinite(ex["band_low"][sel]).any() else None}
    rank = sorted(per, key=lambda s: -per[s]["score"])
    rep = {"video": a.video, "ref": a.ref, "frames": [A, B], "fps": round(fps, 3), "excess": {k: st(ex[k][A:B]) for k in keys}, "ref_resid": {k: st(rr[k][A:B]) for k in keys},
           "colour_flicker": st(fl[A:B]), "colour_offset_lab_mean": [round(float(v), 2) for v in np.nanmean(col_off[A:B], 0)] if np.isfinite(col_off[A:B]).any() else None,
           "sharpness_ratio": round(float(np.nanmean(sharp[A:B, 0]) / np.nanmean(sharp[A:B, 1])), 3) if np.isfinite(sharp[A:B]).any() else None,
           "score": st(score[A:B][np.isfinite(ex["skin_up"][A:B])]), "worst_seconds": [{"s": s, **per[s]} for s in rank[:10]], "per_second": per}
    if a.series: rep["series"] = {**{k: [None if not np.isfinite(v) else round(float(v), 3) for v in ex[k]] for k in keys}, "flicker": [None if not np.isfinite(v) else round(float(v), 3) for v in fl]}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("excess", "colour_flicker", "sharpness_ratio", "score")}), "\nworst:", [(w["s"], w["score"]) for w in rep["worst_seconds"][:10]])


if __name__ == "__main__":
    main()
