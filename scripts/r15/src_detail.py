#!/usr/bin/env python3
"""r15 F2: per-video SOURCE detail transfer onto the LatentSync layer, a CodeFormer replacement. No external restorer, no training,
only this video's own frames.

1. Bank: every master frame with landmarks is aligned to a canonical mouth space (similarity on the mouth corners, at the video's
   native mouth scale, so no resampling loss). Each frame is scored by lip sharpness (mean |Laplacian| of luma in the lip hull),
   openness (inner gap / mouth width) and yaw (nose offset). Per openness bin, the --bank-k sharpest frames are kept.
2. Per generated frame (only where the layer differs from the master = LatentSync frames), the bank entry nearest in lip shape
   (canonical outer + inner lip landmarks) and yaw is chosen, with hysteresis: the previous entry is kept unless the best one is
   clearly better. Switches cross-fade over 3 frames.
3. The entry is warped onto the generated lip shape (thin-plate spline on the 40 lip landmarks plus a fixed border), then
   refined by a clamped DIS flow on low-passed luma. The luma high frequencies are swapped:
       Y += w * conf * mask * (HP(bank) - HP(gen))
   conf = agreement of the low-passed luma (a structure mismatch, e.g. teeth in the bank but not in the generated mouth,
   gets no detail); w = per-frame gain from the sharpness ratio bank / generated, temporally smoothed. If the source is not
   sharper than the layer, w = 0: a no-op on blurry sources. Chroma and low frequencies (shape, opening, motion = SyncNet)
   stay LatentSync's.
    python scripts/r15/src_detail.py --base LS_PRE.mp4 --lm LS.mediapipe.npz --src MASTER_25FPS.mp4 --src-lm MASTER_MEDIAPIPE.npz --out OUT.mp4
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys, time
import numpy as np, cv2
from scipy.interpolate import RBFInterpolator
_LRGB = os.environ.get("LS_LAYER_RGB", "0") == "1"
cv2.setNumThreads(8)
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LIPS = LIPS_OUTER + LIPS_INNER
OPEN_BINS = np.array([0.0, 0.04, 0.10, 0.18, 0.28, 0.40, 9.0])


class Canon:
    """Similarity: mouth corners -> horizontal, mouth width -> MW px, mouth centre -> (CW/2, 0.45 CH)."""
    def __init__(s, MW: float):
        s.MW = float(MW); s.CW = int(round(2.0 * MW)) // 2 * 2; s.CH = int(round(1.6 * MW)) // 2 * 2

    def A(s, p):
        a, b = p[61, :2].astype(np.float64), p[291, :2].astype(np.float64); mw = np.linalg.norm(b - a)
        if not np.isfinite(mw) or mw < 4: return None
        u = (b - a) / mw; v = np.array([-u[1], u[0]]); c = 0.5 * (a + b); k = s.MW / mw
        return np.array([[k * u[0], k * u[1], s.CW / 2 - k * u @ c], [k * v[0], k * v[1], 0.45 * s.CH - k * v @ c]], np.float64)

    def pts(s, A, p): return p[:, :2] @ A[:, :2].T + A[:, 2]

    def crop(s, img, A): return cv2.warpAffine(img, A, (s.CW, s.CH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def lip_mask(q, shape, dil, feather):
    m = np.zeros(shape, np.uint8); cv2.fillPoly(m, [np.round(cv2.convexHull(q[:20].astype(np.float32))[:, 0]).astype(np.int32)], 1)
    if dil > 0: m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1,) * 2))
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(0.8, feather))


def sharp(Y, m): return float((np.abs(cv2.Laplacian(Y.astype(np.float32), cv2.CV_32F)) * m).sum() / max(m.sum(), 1e-6))


def feats(p, A, C):
    q = C.pts(A, p[LIPS]); mw = C.MW
    op = float(np.linalg.norm(p[13, :2] - p[14, :2]) / max(np.linalg.norm(p[61, :2] - p[291, :2]), 1e-6))
    nose = C.pts(A, p[[1]])[0]; yaw = float((nose[0] - C.CW / 2) / mw)
    return q, op, yaw


def main():
    import av
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--src", required=True); ap.add_argument("--src-lm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.8, help="share of the generated HF replaced by the bank HF")
    ap.add_argument("--sigma", type=float, default=0.025, help="HF split, x mouth width")
    ap.add_argument("--bank-k", type=int, default=16); ap.add_argument("--tau", type=float, default=14.0, help="low-pass luma disagreement (levels) -> conf 1/e")
    ap.add_argument("--ratio-lo", type=float, default=1.05); ap.add_argument("--ratio-span", type=float, default=0.35)
    ap.add_argument("--hyst", type=float, default=1.4); ap.add_argument("--dwell", type=int, default=5, help="min frames on one bank entry"); ap.add_argument("--dilate", type=float, default=0.10); ap.add_argument("--max-delta", type=float, default=40)
    ap.add_argument("--crf", type=int, default=10); ap.add_argument("--preset", default="medium")
    a = ap.parse_args(); t0 = time.perf_counter()
    lm = np.load(a.lm)["lm"][..., :2].astype(np.float64); slm = np.load(a.src_lm)["lm"][..., :2].astype(np.float64)
    okL = np.isfinite(lm[:, LIPS + [1]]).all((1, 2)); okS = np.isfinite(slm[:, LIPS + [1]]).all((1, 2))
    MW = float(np.nanmedian(np.linalg.norm(slm[okS, 61] - slm[okS, 291], axis=1))); C = Canon(MW)
    dil = max(1, int(a.dilate * MW)); fe = 0.05 * MW; sh = max(0.8, a.sigma * MW); lp = max(1.0, 0.05 * MW)
    # ---- pass 1: master (bank) + layer (generated crops) ----
    cs, cb = av.open(a.src), av.open(a.base); sb = cb.streams.video[0]; rate = sb.average_rate; cs.streams.video[0].thread_type = cb.streams.video[0].thread_type = "AUTO"
    bankY, bankF = [], []; genY = {}; genF = {}; active = {}; H = W = None
    for k, (fs, fb) in enumerate(zip(cs.decode(video=0), cb.decode(video=0))):
        S = fs.to_ndarray(format="bgr24"); B = fb.to_ndarray(format="bgr24"); H, W = B.shape[:2]
        if k < len(slm) and okS[k]:
            A = C.A(slm[k])
            if A is not None:
                Y = cv2.cvtColor(C.crop(S, A), cv2.COLOR_BGR2GRAY); q, op, yaw = feats(slm[k], A, C)
                bankY.append(Y); bankF.append((k, q, op, yaw, sharp(Y, lip_mask(q, Y.shape, dil, fe))))
        if k < len(lm) and okL[k]:
            A = C.A(lm[k])
            if A is not None:
                Yg = cv2.cvtColor(C.crop(B, A), cv2.COLOR_BGR2GRAY); Ys = cv2.cvtColor(C.crop(S, A), cv2.COLOR_BGR2GRAY); q, op, yaw = feats(lm[k], A, C)
                m = lip_mask(q, Yg.shape, dil, fe)
                active[k] = float((np.abs(Yg.astype(np.float32) - Ys) * m).sum() / max(m.sum(), 1e-6)) > 2.0
                if active[k]: genY[k] = Yg; genF[k] = (A, q, op, yaw, sharp(Yg, m))
    cs.close(); cb.close(); n = k + 1
    # bank: per openness bin the K sharpest
    ops = np.array([f[2] for f in bankF]); shs = np.array([f[4] for f in bankF]); bins = np.digitize(ops, OPEN_BINS) - 1; keep = []
    for b in range(len(OPEN_BINS) - 1):
        idx = np.flatnonzero(bins == b)
        if len(idx): keep += idx[np.argsort(-shs[idx])[:a.bank_k]].tolist()
    keep = np.array(sorted(keep)); KQ = np.stack([bankF[i][1] for i in keep]) / MW; KY = np.array([bankF[i][3] for i in keep]); KB = bins[keep]
    # ---- choose bank entries (lip shape + yaw, hysteresis) ----
    ks = sorted(genY); J = {}; prev = None; cost_of = {}; dwell = 0
    for k in ks:
        A, q, op, yaw, _ = genF[k]; gb = int(np.digitize([op], OPEN_BINS)[0] - 1)
        c = np.abs(KQ - q[None] / MW).sum(2).mean(1) + 0.5 * np.abs(KY - yaw) + 0.02 * np.abs(KB - gb)
        j = int(np.argmin(c))
        if prev is not None and (k - 1) in J and (dwell < a.dwell or c[prev] <= a.hyst * c[j] + 0.004): j = prev
        dwell = dwell + 1 if (prev is not None and j == prev and (k - 1) in J) else 1
        J[k] = j; prev = j; cost_of[k] = float(c[j])
    gen_sharp = np.array([genF[k][4] for k in ks]) if ks else np.zeros(0)
    bank_sharp_sel = np.array([shs[keep[J[k]]] for k in ks]) if ks else np.zeros(0)
    ratio = bank_sharp_sel / np.maximum(gen_sharp, 1e-6); w = np.clip((ratio - a.ratio_lo) / a.ratio_span, 0, 1)
    # temporal smoothing of w over consecutive active frames
    wk = dict(zip(ks, w)); ws = {}
    for k in ks:
        num = den = 0.0
        for d in range(-3, 4):
            if (k + d) in wk: g = np.exp(-0.5 * (d / 1.5) ** 2); num += g * wk[k + d]; den += g
        ws[k] = num / den
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    bord = np.array([[x, y] for x in np.linspace(0, C.CW - 1, 6) for y in (0, C.CH - 1)] + [[x, y] for x in (0, C.CW - 1) for y in np.linspace(0, C.CH - 1, 5)[1:-1]], np.float64)
    gy, gx = np.mgrid[0:C.CH, 0:C.CW].astype(np.float32); step = 8; cgy, cgx = np.mgrid[0:C.CH:step, 0:C.CW:step]; cg = np.stack([cgx.ravel(), cgy.ravel()], 1).astype(np.float64)
    cache = {}
    def delta(k, j):   # canonical luma detail delta for frame k from bank entry j (masked, conf-weighted, before w)
        key = (k, j)
        if key in cache: return cache[key]
        A, q, op, yaw, _ = genF[k]; bi = keep[j]; qb = bankF[bi][1]; Yb = bankY[bi].astype(np.float32); Yg = genY[k].astype(np.float32)
        src = np.vstack([qb, bord]); dst = np.vstack([q, bord])   # for every dst (generated) pixel: where to sample the bank
        rbf = RBFInterpolator(dst, src - dst, kernel="thin_plate_spline", smoothing=1.0); d = rbf(cg).reshape(cgy.shape + (2,)).astype(np.float32)
        d = cv2.resize(d, (C.CW, C.CH), interpolation=cv2.INTER_LINEAR); Ybw = cv2.remap(Yb, gx + d[..., 0], gy + d[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        Lg, Lb = cv2.GaussianBlur(Yg, (0, 0), 1.5), cv2.GaussianBlur(Ybw, (0, 0), 1.5)
        F = dis.calc(np.clip(Lg, 0, 255).astype(np.uint8), np.clip(Lb, 0, 255).astype(np.uint8), None); mag = np.linalg.norm(F, axis=2, keepdims=True); cap = 0.04 * MW
        F = F * np.minimum(1.0, cap / np.maximum(mag, 1e-6)); Yba = cv2.remap(Ybw, gx + F[..., 0], gy + F[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        conf = np.exp(-(np.abs(cv2.GaussianBlur(Yg, (0, 0), lp) - cv2.GaussianBlur(Yba, (0, 0), lp)) / a.tau) ** 2)
        hp_b = Yba - cv2.GaussianBlur(Yba, (0, 0), sh); hp_g = Yg - cv2.GaussianBlur(Yg, (0, 0), sh)
        out = np.clip(a.alpha * (hp_b - hp_g), -a.max_delta, a.max_delta) * conf * lip_mask(q, Yg.shape, dil, fe)
        if len(cache) > 64: cache.clear()
        cache[key] = (out, float(conf[lip_mask(q, Yg.shape, 0, 1) > 0.5].mean())); return cache[key]
    # ---- pass 2: apply + encode ----
    cb = av.open(a.base); cb.streams.video[0].thread_type = "AUTO"
    o = av.open(a.out, "w"); st = o.add_stream("libx264rgb" if _LRGB else "libx264", rate=rate); st.width, st.height = W, H; st.pix_fmt = "bgr24" if _LRGB else "yuv420p"
    st.options = {"qp": "0", "preset": a.preset} if _LRGB else {"crf": str(a.crf), "preset": a.preset}
    applied = 0; confs = []; switches = 0; last_switch = -99
    for k, fb in enumerate(cb.decode(video=0)):
        img = fb.to_ndarray(format="bgr24")
        if k in genY and ws[k] > 0.01:
            j = J[k]
            if (k - 1) in J and J[k - 1] != j: switches += 1; last_switch = k; jprev = J[k - 1]
            D, cm = delta(k, j)
            if 0 <= k - last_switch < 3 and (k - 1) in genY:   # cross-fade from the previous entry
                t = (k - last_switch + 1) / 3.0; D = t * D + (1 - t) * delta(k, jprev)[0]
            D = ws[k] * D; confs.append(cm)
            A = genF[k][0]; Ai = cv2.invertAffineTransform(A); corners = np.array([[0, 0], [C.CW, 0], [0, C.CH], [C.CW, C.CH]], np.float64) @ Ai[:, :2].T + Ai[:, 2]
            x0, y0 = np.maximum(0, np.floor(corners.min(0)).astype(int)); x1, y1 = np.minimum([W, H], np.ceil(corners.max(0)).astype(int) + 1)
            if x1 - x0 > 8 and y1 - y0 > 8:
                Ab = A.copy(); Ab[:, 2] += Ab[:, :2] @ np.array([x0, y0], np.float64)   # box pixel -> canonical
                Di = cv2.warpAffine(D, Ab, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                ycc = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2YCrCb).astype(np.float32); ycc[..., 0] = np.clip(ycc[..., 0] + Di, 0, 255)
                img = img.copy(); img[y0:y1, x0:x1] = cv2.cvtColor(ycc.round().astype(np.uint8), cv2.COLOR_YCrCb2BGR); applied += 1
        for pk in st.encode(av.VideoFrame.from_ndarray(img, format="bgr24")): o.mux(pk)
    for pk in st.encode(): o.mux(pk)
    o.close(); cb.close()
    rep = {"frames": n, "ls_frames": len(ks), "applied": applied, "mouth_width_px": round(MW, 1), "bank": {"frames": len(bankF), "kept": len(keep), "per_bin": np.bincount(KB, minlength=len(OPEN_BINS) - 1).tolist()},
           "sharp_ratio_bank_over_gen": {"p10": round(float(np.percentile(ratio, 10)), 3), "p50": round(float(np.median(ratio)), 3), "p90": round(float(np.percentile(ratio, 90)), 3)} if len(ratio) else None,
           "w_mean": round(float(np.mean(list(ws.values()))), 3) if ws else 0.0, "conf_mean": round(float(np.mean(confs)), 3) if confs else None,
           "switches": switches, "seconds": round(time.perf_counter() - t0, 1), "args": vars(a)}
    pathlib.Path(a.out).with_suffix(".sdt.json").write_text(json.dumps(rep, indent=1)); print("== src_detail", json.dumps({k: v for k, v in rep.items() if k != "args"}), flush=True)


if __name__ == "__main__":
    main()
