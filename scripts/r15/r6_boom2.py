#!/usr/bin/env python3
"""r6 m03 'explosion' (user: 'взрыв во рту' at m03 ~8.6 s, student lips swollen / doubled, production fine): candidate per-frame signals
on the SOURCE timeline (source landmarks), finals resized to the source:
  pop1  = explode.pop_at (single-frame pop, F's)
  popR  = |f_k - (w_{k-R} + w_{k+R}) / 2| in the lip hull (multi-frame pop, R = --r)
  tex   = lip-hull Laplacian variance / its +-1 s running median (texture burst)
  dref  = mean |f_k - ref_k| in the lip hull vs a reference final (--ref, e.g. q2u): student-vs-teacher divergence
prints per final the top-5 times of each signal and the value in --at (s) vs the 95th percentile elsewhere (the margin)."""
import argparse, sys, json
import numpy as np, cv2
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import explode as X


def hull_mask(shape, p, dil):
    q = p[X.LIPS, :2]; iod = float(np.linalg.norm(p[33, :2] - p[263, :2])); m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(cv2.convexHull(q.astype(np.float32))[:, 0]).astype(np.int32)], 1)
    if dil > 0: m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(dil * iod) + 1,) * 2))
    return m > 0


def popR(F, lm, k, R):
    if k - R < 0 or k + R >= len(F): return np.nan
    p = lm[k]
    if not np.isfinite(p[X.LIPS + [33, 263], :2]).all(): return np.nan
    iod = float(np.linalg.norm(p[33, :2] - p[263, :2])); q = p[X.LIPS, :2]; r = int(0.6 * iod)
    x0, y0 = np.maximum(q.min(0).astype(int) - r, 0); x1, y1 = q.max(0).astype(int) + r
    A, B, C = [np.ascontiguousarray(F[j][y0:y1, x0:x1]) for j in (k - R, k, k + R)]
    gy, gx = np.mgrid[0:B.shape[0], 0:B.shape[1]].astype(np.float32)
    f = X.dis.calc(B, A, None); g = X.dis.calc(B, C, None)
    wa = cv2.remap(A, gx + f[..., 0], gy + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    wc = cv2.remap(C, gx + g[..., 0], gy + g[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    m = hull_mask(B.shape, p - [x0, y0, 0][:p.shape[1]], 0.3)
    return float(np.abs(B.astype(np.float32) - 0.5 * (wa + wc))[m].mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--src-lm", required=True)
    ap.add_argument("--t0", type=float, default=0); ap.add_argument("--t1", type=float, default=1e9); ap.add_argument("--final", action="append", required=True)
    ap.add_argument("--ref"); ap.add_argument("--r", type=int, default=3); ap.add_argument("--at", type=float, nargs=2, default=[8.4, 8.9]); ap.add_argument("--json"); a = ap.parse_args()
    slm = np.load(a.src_lm)["lm"].astype(np.float64); cap = cv2.VideoCapture(a.src); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    S = X.read_all(a.src); n = min(len(S), len(slm)); k0, k1 = max(int(a.t0 * fps), 3), min(int(a.t1 * fps), n - 4)
    def load(v):
        F = X.read_all(v); return [cv2.resize(x, (S[0].shape[1], S[0].shape[0])) for x in F] if F[0].shape != S[0].shape else F
    Rf = load(a.ref) if a.ref else None
    def sig(F):
        ks = range(k0, k1); out = {"pop1": [], "popR": [], "tex": [], "dref": []}; lv = []
        for k in ks:
            p = slm[k]
            if not np.isfinite(p[X.LIPS + [33, 263], :2]).all():
                for kk in out: out[kk].append(np.nan)
                lv.append(np.nan); continue
            out["pop1"].append(X.pop_at(F, slm, k)); out["popR"].append(popR(F, slm, k, a.r))
            m = hull_mask(F[k].shape, p, 0.0); lap = cv2.Laplacian(F[k], cv2.CV_32F); lv.append(float(lap[m].var()))
            out["dref"].append(float(np.abs(F[k].astype(np.float32) - Rf[k].astype(np.float32))[hull_mask(F[k].shape, p, 0.2)].mean()) if Rf is not None and k < len(Rf) else np.nan)
        lv = np.array(lv); w = int(fps)
        out["tex"] = [lv[i] / max(1e-6, np.nanmedian(lv[max(0, i - w):i + w + 1])) for i in range(len(lv))]
        return {kk: np.array(v, float) for kk, v in out.items()}
    t = np.arange(k0, k1) / fps; inw = (t >= a.at[0]) & (t <= a.at[1]); res = {}
    for spec in a.final:
        lab, v = spec.split("=", 1); s = sig(load(v)); res[lab] = {kk: np.round(vv, 3).tolist() for kk, vv in s.items()}; line = []
        for kk, vv in s.items():
            if np.all(np.isnan(vv)): continue
            top = np.argsort(-np.nan_to_num(vv, nan=-1e9))[:5]; inmax = np.nanmax(vv[inw]) if inw.any() else np.nan; out95 = np.nanpercentile(vv[~inw], 95); outmax = np.nanmax(vv[~inw])
            line.append(f"{kk}: in[{a.at[0]}-{a.at[1]}] max {inmax:.2f} | outside p95 {out95:.2f} max {outmax:.2f} | top " + " ".join(f"{t[i]:.2f}" for i in top))
        print(f"== {lab}\n   " + "\n   ".join(line), flush=True)
    if a.json: res["t"] = t.tolist(); open(a.json, "w").write(json.dumps(res))


if __name__ == "__main__":
    main()
