#!/usr/bin/env python3
"""r15 F2: lip-vs-skin contrast of a layer as a fraction of the SOURCE at the same master frame (the "lips vanish" test).

Measure (r15 T scripts/r15/lip_vanish.py): on the lip ring (outer lip polygon minus the inner) vs a skin ring around the mouth,
  a_con = median a* (CIELAB) ring - skin        (lips redder than skin)
  dL    = median L skin - ring                  (lips darker than skin)
  ratio = min(a_con_layer / max(a_con_src, 1), dL_layer / max(dL_src, 2))    1 = as visible as the source, ~0 = the lips are gone
counted only where the source has visible lips (a_con_src > 4 or dL_src > 4).
Library: measure(), contrast(); r8_fallback.py --thr-lipcon uses it. CLI: dip runs of one or more layers (25 fps master timeline):
    python scripts/r15/lipcon.py --work WORK --layer label=LAYER.mp4 [--layer label=FINAL.mp4@LAYER.mp4 --timing T.json] [...] [--thr 0.45] [--json out]
"""
import argparse, json, pathlib
import numpy as np, cv2

OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


def measure(img, lm):
    """-> (a_con, dL) for one BGR frame and its 478 x 2(3) pixel landmarks; NaN when the mouth is missing"""
    if lm is None or not np.isfinite(lm[OUTER + INNER, :2]).all(): return np.nan, np.nan
    o = lm[OUTER, :2].astype(np.int32); i = lm[INNER, :2].astype(np.int32)
    x0, y0 = o.min(0); x1, y1 = o.max(0); mw = max(8, x1 - x0); pad = int(0.6 * mw)
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad); X1, Y1 = min(img.shape[1], x1 + pad), min(img.shape[0], y1 + pad)
    crop = img[Y0:Y1, X0:X1]; off = np.array([X0, Y0])
    if crop.size == 0: return np.nan, np.nan
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    mo = np.zeros(crop.shape[:2], np.uint8); cv2.fillPoly(mo, [o - off], 1)
    mi = np.zeros_like(mo); cv2.fillPoly(mi, [i - off], 1)
    ring = (mo == 1) & (mi == 0)
    k1 = max(3, int(0.08 * mw)) | 1; k2 = max(5, int(0.30 * mw)) | 1
    skin = (cv2.dilate(mo, np.ones((k2, k2), np.uint8)) == 1) & (cv2.dilate(mo, np.ones((k1, k1), np.uint8)) == 0)
    if ring.sum() < 20 or skin.sum() < 20: return np.nan, np.nan
    return (float(np.median(lab[..., 1][ring]) - np.median(lab[..., 1][skin])), float(np.median(lab[..., 0][skin]) - np.median(lab[..., 0][ring])))


def contrast(lay, src):
    """per-frame ratio from (n, 2) layer / source (a_con, dL) arrays; NaN where the source lips are not visible"""
    r = np.minimum(lay[:, 0] / np.maximum(src[:, 0], 1.0), lay[:, 1] / np.maximum(src[:, 1], 2.0))
    return np.where((src[:, 0] > 4) | (src[:, 1] > 4), r, np.nan)


def series(video, lm, n):
    cap = cv2.VideoCapture(str(video)); out = np.full((n, 2), np.nan)
    for k in range(n):
        ok, f = cap.read()
        if not ok: break
        out[k] = measure(f, lm[k])
    cap.release(); return out


def runs(r, thr, mask):
    d = np.nan_to_num(r, nan=9) < thr; d &= mask; out, k = [], 0
    while k < len(d):
        if d[k]:
            e = k
            while e + 1 < len(d) and d[e + 1]: e += 1
            w = min(range(k, e + 1), key=lambda q: r[q]); out.append([round(k / 25, 2), round((e + 1) / 25, 2), round(float(r[w]), 2)]); k = e + 1
        else: k += 1
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--layer", action="append", required=True)
    ap.add_argument("--thr", type=float, default=0.45); ap.add_argument("--src-lm"); ap.add_argument("--json")
    ap.add_argument("--timing", help="<id>_master_timing_F.json (its 'j' = the source frame of each master frame), for FINAL=VIDEO@LAYER specs")
    a = ap.parse_args(); W = pathlib.Path(a.work)
    slm = np.load(a.src_lm or W / "video/master_mediapipe.npz")["lm"]
    man = json.loads(next(W.glob("*_25fps.manifest.json")).read_text()); n = len(slm); ls = np.zeros(n, bool)
    for s in man["lipsync"]["segments"]:
        if s["action"] == "LATENT_SYNC": ls[s["start_frame"]:min(n, s["end_frame"])] = True
    src = series(W / "video/master_25fps.mp4", slm, n); res = {}
    c_ = cv2.VideoCapture(str(W / "video/master_25fps.mp4")); slm_wh = (c_.get(cv2.CAP_PROP_FRAME_WIDTH), c_.get(cv2.CAP_PROP_FRAME_HEIGHT)); c_.release()
    J = np.array(json.load(open(a.timing))["j"]) if a.timing else None
    for spec in a.layer:
        lab, v = spec.split("=", 1); lmp = None
        if "@" in v: v, lmp = v.split("@", 1)   # a FINAL video (source timeline) @ the layer whose landmarks it shows: master frame k = final frame J[k]
        v = pathlib.Path(v); lm = np.load(pathlib.Path(lmp).with_suffix(".mediapipe.npz") if lmp else v.with_suffix(".mediapipe.npz"))["lm"]
        if lmp:
            cap = cv2.VideoCapture(str(v)); idx, f, m = -1, None, []   # J is non-decreasing: stream the final, keep the current frame only
            for k in range(min(n, len(lm))):
                while idx < J[k]:
                    ok, g = cap.read()
                    if not ok: break
                    idx += 1; f = cv2.resize(g, (int(slm_wh[0]), int(slm_wh[1]))) if g.shape[1] != slm_wh[0] else g
                m.append(measure(f, lm[k]) if f is not None else (np.nan, np.nan))
            cap.release(); m = np.array(m)
        else: m = series(v, lm, min(n, len(lm)))
        r = contrast(m, src[:len(m)]); r = np.pad(r, (0, n - len(r)), constant_values=np.nan)
        rr = runs(r, a.thr, ls); res[lab] = {"median": round(float(np.nanmedian(r[ls])), 3), "p5": round(float(np.nanpercentile(r[ls], 5)), 3),
                                             "dip_frames": int(sum(round(25 * (t1 - t0)) for t0, t1, _ in rr)), "runs": rr}
        print("== lipcon", lab, json.dumps(res[lab]), flush=True)
    if a.json: pathlib.Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
