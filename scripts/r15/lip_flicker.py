#!/usr/bin/env python3
"""r15 F3: LIP colour / brightness flicker ("lip blinking") of a final, with the source's own value as the floor.
On the lip ring (outer lip polygon minus the inner one, eroded 1 px) of EACH video's own mouth:
  final  at source frame k: the layer landmarks of the master frame shown there (master_timing_F "j", nearest), on the final picture
  source at source frame k: the source landmarks (source_lm_F)
mean L and a* (CIELAB, 8-bit scale) per frame; dL, da = frame-to-frame change. Lip colour legitimately changes with the mouth shape,
so a BLINK is a jump at a steady shape: |dL| > --thr-l or |da| > --thr-a while the opening changes < --op-thr IOD.
Printed: blink events and per 10 s for the final and the source, excess = final - source, mean |dL| / |da| at steady shape, top blink times.
    python scripts/r15/lip_flicker.py --video FINAL --lm LAYER.mediapipe.npz --timing master_timing_F.json --master master_25fps.mp4
                                      --src SOURCE --src-lm source_lm_F.npz --label x [--json out.jsonl]"""
import argparse, json
import numpy as np, cv2
OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


def stats(img, p):
    if p is None or not np.isfinite(p[OUTER + INNER + [33, 263], :2]).all(): return np.nan, np.nan, np.nan, np.nan
    o = np.round(p[OUTER, :2]).astype(np.int32); x0, y0 = np.maximum(o.min(0) - 4, 0); x1, y1 = o.max(0) + 4
    c = img[y0:y1, x0:x1]
    if c.size == 0: return np.nan, np.nan, np.nan, np.nan
    m = np.zeros(c.shape[:2], np.uint8); cv2.fillPoly(m, [o - [x0, y0]], 1); cv2.fillPoly(m, [np.round(p[INNER, :2]).astype(np.int32) - [x0, y0]], 0)
    m = cv2.erode(m, np.ones((3, 3), np.uint8)) > 0
    if m.sum() < 20: return np.nan, np.nan, np.nan, np.nan
    lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB).astype(np.float32)[m]
    op = np.linalg.norm(p[13, :2] - p[14, :2]) / max(np.linalg.norm(p[33, :2] - p[263, :2]), 1)
    lv = float(cv2.Laplacian(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.CV_32F)[m].var())
    return float(lab[:, 0].mean()), float(lab[:, 1].mean()), float(op), lv


def blinks(X, thr_l, thr_a, op_thr, fps):
    d = np.diff(X, axis=0); ok = np.isfinite(d).all(1) & (np.abs(d[:, 2]) < op_thr)
    dl, da = np.abs(d[:, 0]), np.abs(d[:, 1]); ev = ok & ((dl > thr_l) | (da > thr_a))
    return ev, ok, dl, da


def main():
    ap = argparse.ArgumentParser(); [ap.add_argument("--" + k, required=True) for k in ("video", "lm", "timing", "master", "src", "src-lm")]
    ap.add_argument("--label", default=""); ap.add_argument("--thr-l", type=float, default=5.0); ap.add_argument("--thr-a", type=float, default=2.5)
    ap.add_argument("--op-thr", type=float, default=0.03); ap.add_argument("--json")
    a = ap.parse_args(); llm = np.load(a.lm)["lm"].astype(np.float64); slm = np.load(a.src_lm)["lm"].astype(np.float64)
    J = np.array(json.load(open(a.timing))["j"]); cm = cv2.VideoCapture(a.master); mw, mh = int(cm.get(3)), int(cm.get(4)); cm.release()
    cv, cs = cv2.VideoCapture(a.video), cv2.VideoCapture(a.src); fps = cs.get(cv2.CAP_PROP_FPS) or 30.0
    F, S = [], []; k = 0
    while True:
        okv, fv = cv.read(); oks, fs = cs.read()
        if not (okv and oks): break
        m = int(np.clip(np.searchsorted(J, k), 0, len(llm) - 1)); m = m if abs(J[min(m, len(J) - 1)] - k) <= 1 else None
        g = cv2.resize(fv, (mw, mh)) if (fv.shape[1], fv.shape[0]) != (mw, mh) else fv
        F.append(stats(g, llm[m] if m is not None else None)); S.append(stats(fs, slm[k] if k < len(slm) else None)); k += 1
    F, S = np.array(F), np.array(S); dur = len(F) / fps
    eF, okF, dlF, daF = blinks(F, a.thr_l, a.thr_a, a.op_thr, fps); eS, okS, dlS, daS = blinks(S, a.thr_l, a.thr_a, a.op_thr, fps)
    # W-style (r15 W per-stage table): all frames with lips on both sides, any shape: excess mean |dL| over the source, and the count of
    # |dL| above the SOURCE's p99 |dL| (the source itself has ~1 % by construction)
    vF, vS = np.isfinite(np.diff(F[:, 0])), np.isfinite(np.diff(S[:, 0])); aF, aS = np.abs(np.diff(F[:, 0])), np.abs(np.diff(S[:, 0]))
    p99 = float(np.percentile(aS[vS], 99)) if vS.any() else np.nan
    # r15 audit item 1: the 25 -> 30 SHARPNESS pulse: lip-ring Laplacian variance (/ its +-15-frame running median) on the source frames that
    # ARE a master frame (|mu - round(mu)| < 0.08) vs the interpolated ones (> 0.25); 1.0 = no pulse (the source's own ratio is the floor)
    pulse = {}
    mu = np.load(a.src_lm)["mu"] if "mu" in np.load(a.src_lm).files else None
    if mu is not None:
        n_ = min(len(mu), len(F)); dist = np.abs(mu[:n_] - np.round(mu[:n_]))
        for nm, X in (("final", F), ("source", S)):
            lv = X[:n_, 3]; rm = np.array([np.nanmedian(lv[max(0, k - 15):k + 16]) for k in range(n_)]); lvn = lv / np.maximum(rm, 1e-6)
            r0, r1 = np.nanmedian(lvn[dist < 0.08]), np.nanmedian(lvn[dist > 0.25]); pulse[nm] = round(float(r0 / r1), 3)
    sc = np.where(eF, np.maximum(dlF / a.thr_l, daF / a.thr_a), 0); top = [i for i in np.argsort(-sc)[:12] if sc[i] > 0]
    r = {"label": a.label, "frames": int(len(F)), "blinks": int(eF.sum()), "blinks_per_10s": round(10 * eF.sum() / dur, 2), "src_blinks": int(eS.sum()),
         "excess_per_10s": round(10 * (eF.sum() - eS.sum()) / dur, 2),
         "W_excess_dL": round(float(aF[vF].mean() - aS[vS].mean()), 3), "W_jumps_p99": int((aF[vF] > p99).sum()), "W_src_jumps_p99": int((aS[vS] > p99).sum()), "src_p99_dL": round(p99, 2), "sharp_pulse_raw_vs_interp": pulse,
         "dL_steady": round(float(dlF[okF].mean()), 3), "da_steady": round(float(daF[okF].mean()), 3),
         "src_dL_steady": round(float(dlS[okS].mean()), 3), "src_da_steady": round(float(daS[okS].mean()), 3),
         "top": [[round((i + 1) / fps, 2), round(float(dlF[i]), 1), round(float(daF[i]), 1)] for i in sorted(top)]}
    print("== lip_flicker", json.dumps(r), flush=True)
    if a.json: open(a.json, "a").write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
