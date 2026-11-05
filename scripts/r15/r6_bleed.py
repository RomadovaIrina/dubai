#!/usr/bin/env python3
"""r6 m03 'explosion' = the composite, not the generator: per source frame k, the final's lip region vs the final layer at master frame
round(mu[k]) (the composite should show the layer there). bleed_k = share of lip-hull pixels (source landmarks, + 0.15 IOD) where the
final is > --dark darker than the layer (the source's open mouth / teeth showing through a closed generated mouth), and the source
mouth opening vs the layer's. Spikes = frames where the composite shows something the generator never drew.
    python bleed.py --src-lm source_lm_F.npz --final q2u=F.mp4 --layer q2u=L.mp4 [--final ... --layer ...] [--json o.json]"""
import argparse, json
import numpy as np, cv2
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


def read(p):
    c = cv2.VideoCapture(p); out = []
    while True:
        ok, f = c.read()
        if not ok: break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src-lm", required=True); ap.add_argument("--final", action="append", required=True)
    ap.add_argument("--layer", action="append", required=True); ap.add_argument("--dark", type=float, default=25.0); ap.add_argument("--json"); a = ap.parse_args()
    z = np.load(a.src_lm); slm = z["lm"].astype(np.float64); mu = z["mu"]; res = {}
    for fs, ls in zip(a.final, a.layer):
        lab, fv = fs.split("=", 1); _, lv = ls.split("=", 1); F = read(fv); L = read(lv)
        if L[0].shape != F[0].shape: L = [cv2.resize(x, (F[0].shape[1], F[0].shape[0])) for x in L]
        n = min(len(F), len(slm), len(mu)); b = np.full(n, np.nan)
        for k in range(n):
            p = slm[k]; m_ = int(round(mu[k]))
            if m_ >= len(L) or not np.isfinite(p[LIPS + [33, 263], :2]).all(): continue
            iod = float(np.linalg.norm(p[33, :2] - p[263, :2])); m = np.zeros(F[k].shape, np.uint8)
            cv2.fillPoly(m, [np.round(cv2.convexHull(p[LIPS, :2].astype(np.float32))[:, 0]).astype(np.int32)], 1)
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.15 * iod) + 1,) * 2)) > 0
            d = L[m_].astype(np.float32)[m] - F[k].astype(np.float32)[m]
            b[k] = float((d - np.median(d) > a.dark).mean())   # minus the median: the composite's global colour match is not a bleed
        top = np.argsort(-np.nan_to_num(b))[:8]
        print(f"{lab}: bleed p50 {np.nanmedian(b):.4f} p99 {np.nanpercentile(b, 99):.4f} max {np.nanmax(b):.4f}; frames > 0.05: {int(np.nansum(b > 0.05))}; top "
              + " ".join(f"{k / 29.97:.2f}s:{b[k]:.3f}" for k in top), flush=True)
        res[lab] = np.round(b, 4).tolist()
    if a.json: open(a.json, "w").write(json.dumps(res))


if __name__ == "__main__":
    main()
