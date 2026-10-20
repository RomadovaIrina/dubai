#!/usr/bin/env python3
"""Round 4 V gate metrics on a source-fps OUTPUT vs its SOURCE (same frame count), per frame with the source's MediaPipe landmarks:
  colour   mean Lab difference (out - src) of the lower face, LEFT and RIGHT half separately (v5 / 04: "half of the face goes dark"):
           p99 |dL|, worst 1 s rolling mean |dL| and |dab| (chroma), frames with |dL| > 6 grey levels
  jaw      (a) jaw-contour landmark jitter: MediaPipe on the output, jaw points relative to the nose bridge / inter-ocular distance,
               mean second difference out / src (1.0 = as steady as the source; the "chin wave" pushes it up)
           (b) below-jaw band: mean |out - src| in a band from the jaw contour down to +0.3 IOD (second / third chin; the generated
               region must not reach it: untouched = encoder noise)
    python scripts/r4/r4_face_metrics.py --video OUT.mp4 --src SRC.mp4 --src-lm SRC_lm.npz --out x.json [--frames a:b]"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]
CHIN = [17, 314, 405, 321, 375, 291, 287, 432, 434, 430, 394, 379, 378, 400, 377, 152, 148, 176, 149, 150, 169, 210, 214, 212, 57, 61, 146, 91, 181, 84]   # lower lip -> jaw
JAW = [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361]
NOSE_BRIDGE, NOSE_TIP, EYE_L, EYE_R = 168, 1, 33, 263


def rolling_max(x, w):
    x = np.asarray(x, float); ok = np.isfinite(x)
    if ok.sum() < w: return float(np.nanmax(x)) if ok.any() else None
    c = np.convolve(np.where(ok, x, 0), np.ones(w), "valid"); n = np.convolve(ok.astype(float), np.ones(w), "valid")
    m = np.where(n >= w // 2, c / np.maximum(n, 1), np.nan); return float(np.nanmax(m))


def jaw_rel(lm):
    """(T, P, 3) pixel landmarks -> (T, len(JAW), 2) jaw points relative to the nose bridge, in inter-ocular units."""
    iod = np.linalg.norm(lm[:, EYE_L, :2] - lm[:, EYE_R, :2], axis=1)[:, None, None]
    return (lm[:, JAW, :2] - lm[:, NOSE_BRIDGE, None, :2]) / iod


def accel(x):
    a = np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=2).mean(1); return np.concatenate([[np.nan], a, [np.nan]])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--src", required=True); ap.add_argument("--src-lm", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--frames", default=None); ap.add_argument("--no-jaw-lm", action="store_true"); a = ap.parse_args()
    import av, cv2
    lm = np.load(a.src_lm)["lm"]; T = len(lm)
    f0, f1 = (map(int, a.frames.split(":")) if a.frames else (0, T))
    cv, cs = av.open(a.video), av.open(a.src); fps = float(cs.streams.video[0].average_rate)
    rows = []; prev = None
    for k, (fo, fs) in enumerate(zip(cv.decode(video=0), cs.decode(video=0))):
        if k >= f1: break
        if k < f0 or not np.isfinite(lm[k, :, :2]).all(): rows.append(None); continue
        o, s = fo.to_ndarray(format="bgr24"), fs.to_ndarray(format="bgr24"); H, W = s.shape[:2]; p = lm[k, :, :2]
        x0, y0 = np.maximum(np.floor(p.min(0)).astype(int) - 40, 0); x1, y1 = np.minimum(np.ceil(p.max(0)).astype(int) + 60, [W, H])
        oc, sc = o[y0:y1, x0:x1], s[y0:y1, x0:x1]; q = p - [x0, y0]
        Lo, Ls = cv2.cvtColor(oc, cv2.COLOR_BGR2LAB).astype(np.float32), cv2.cvtColor(sc, cv2.COLOR_BGR2LAB).astype(np.float32)
        face = np.zeros(oc.shape[:2], np.uint8); cv2.fillPoly(face, [np.round(q[FACE_OVAL]).astype(np.int32)], 1)
        yy, xx = np.mgrid[0:oc.shape[0], 0:oc.shape[1]]; low = face.astype(bool) & (yy >= q[NOSE_TIP, 1] - 0.15 * np.linalg.norm(q[EYE_L] - q[EYE_R]))
        r = {"k": k}
        for name, m in (("left", low & (xx < q[NOSE_TIP, 0])), ("right", low & (xx >= q[NOSE_TIP, 0]))):
            if m.sum() < 50: r[name] = None; continue
            d = (Lo[m] - Ls[m]).mean(0); r[name] = [float(d[0]) * 100 / 255, float(np.hypot(d[1], d[2]))]   # dL in L* units, dab chroma
        # below-jaw band: jaw contour shifted down by 0..0.3 IOD, outside the face oval
        iod = np.linalg.norm(q[EYE_L] - q[EYE_R]); jaw = np.round(q[JAW]).astype(np.int32)
        band = np.zeros_like(face); cv2.fillPoly(band, [np.concatenate([jaw, (jaw + [0, int(0.3 * iod)])[::-1]])], 1); band &= (1 - face)
        r["below_jaw"] = float(np.abs(oc.astype(np.float32) - sc.astype(np.float32)).mean(2)[band.astype(bool)].mean()) if band.sum() > 50 else None
        # chin flicker: frame-to-frame |change| in the chin area (below the lower lip, inside the jaw) of OUT vs the same for the SOURCE
        # (head motion is in both; a regenerated chin that "waves" adds its own change)
        cm = np.zeros(s.shape[:2], np.uint8); cv2.fillPoly(cm, [np.round(p[CHIN]).astype(np.int32)], 1)   # full-frame coordinates (the crop moves)
        go, gs = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY).astype(np.float32), cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None and prev[0] == k - 1 and prev[1].shape == go.shape and cm.sum() > 50:
            b = cm.astype(bool); r["chin_dt"] = [float(np.abs(go - prev[1])[b].mean()), float(np.abs(gs - prev[2])[b].mean())]
        prev = (k, go, gs)
        rows.append(r)
    cv.close(); cs.close()
    ok = [r for r in rows if r]; res = {"video": a.video, "src": a.src, "frames": [f0, f1], "frames_with_face": len(ok)}
    w = max(int(round(fps)), 1)
    for name in ("left", "right"):
        dL = np.array([r[name][0] if r and r.get(name) else np.nan for r in rows]); dC = np.array([r[name][1] if r and r.get(name) else np.nan for r in rows])
        res[f"colour_{name}"] = {"p99_abs_dL": round(float(np.nanpercentile(np.abs(dL), 99)), 2) if np.isfinite(dL).any() else None,
                                 "worst_1s_abs_dL": round(rolling_max(np.abs(dL), w) or 0, 2), "worst_1s_dab": round(rolling_max(dC, w) or 0, 2),
                                 "frames_abs_dL_gt6": int(np.nansum(np.abs(dL) > 6)), "mean_dL": round(float(np.nanmean(dL)), 2) if np.isfinite(dL).any() else None}
    bj = np.array([r["below_jaw"] if r and r.get("below_jaw") is not None else np.nan for r in rows])
    res["below_jaw"] = {"mean": round(float(np.nanmean(bj)), 3), "p95": round(float(np.nanpercentile(bj, 95)), 3), "worst_1s": round(rolling_max(bj, w) or 0, 3)}
    cd = np.array([r["chin_dt"] if r and r.get("chin_dt") else [np.nan, np.nan] for r in rows])
    rat = cd[:, 0] / np.maximum(cd[:, 1], 0.3)
    res["chin_flicker"] = {"ratio_out_over_src": round(float(np.nanmean(cd[:, 0]) / max(np.nanmean(cd[:, 1]), 1e-6)), 3) if np.isfinite(cd).any() else None,
                           "worst_1s_ratio": round(rolling_max(rat, w) or 0, 3), "p95_frame_ratio": round(float(np.nanpercentile(rat, 95)), 3) if np.isfinite(rat).any() else None}
    res["colour_worst_1s_abs_dL"] = max(res["colour_left"]["worst_1s_abs_dL"], res["colour_right"]["worst_1s_abs_dL"])
    if not a.no_jaw_lm:
        import clean_face_stack as cfs
        npz = pathlib.Path(a.out).with_suffix(".out_mp.npz"); lo = np.load(cfs.run_mediapipe(pathlib.Path(a.video), npz))["lm"]
        n = min(len(lo), T); ao, as_ = accel(jaw_rel(lo[:n])), accel(jaw_rel(lm[:n])); sel = np.zeros(n, bool); sel[f0:f1] = True
        ok = sel & np.isfinite(ao) & np.isfinite(as_)
        res["jaw_jitter"] = {"ratio_out_over_src": round(float(ao[ok].mean() / max(as_[ok].mean(), 1e-9)), 3) if ok.any() else None,
                             "out_mean": round(float(ao[ok].mean()), 5) if ok.any() else None, "src_mean": round(float(as_[ok].mean()), 5) if ok.any() else None,
                             "worst_1s_excess": round(rolling_max(np.where(ok, ao - as_, np.nan), w) or 0, 5), "frames": int(ok.sum())}
    per = {"below_jaw": bj, "chin_dt_out": cd[:, 0], "chin_dt_src": cd[:, 1], **{f"dL_{n}": np.array([r[n][0] if r and r.get(n) else np.nan for r in rows]) for n in ("left", "right")}}
    if not a.no_jaw_lm: per.update(jaw_acc_out=ao, jaw_acc_src=as_)
    np.savez_compressed(pathlib.Path(a.out).with_suffix(".perframe.npz"), **per)   # per-frame series (review sheets, LS vs pass-through checks)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("video", "src")}))


if __name__ == "__main__":
    main()
