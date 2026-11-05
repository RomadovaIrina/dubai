#!/usr/bin/env python3
"""mouth crops, source frames k0..k1 (every --step): rows = frames, columns = the given finals (source timeline) + the given 25 fps
layers (master frame round(mu[k]) via the source landmarks' mu), labelled; one jpg.
    python boom_tiles.py --src-lm source_lm_F.npz --k0 252 --k1 264 --final q2u=A.mp4 --final r6=B.mp4 --layer r6pre=L.mp4 --out o.jpg"""
import argparse, numpy as np, cv2


def frames(p, idx):
    c = cv2.VideoCapture(p); out = {}; i = 0; want = set(idx)
    while i <= max(idx):
        ok, f = c.read()
        if not ok: break
        if i in want: out[i] = f
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src-lm", required=True); ap.add_argument("--k0", type=int); ap.add_argument("--k1", type=int); ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--final", action="append", default=[]); ap.add_argument("--layer", action="append", default=[]); ap.add_argument("--layer-lm", action="append", default=[])
    ap.add_argument("--out", required=True); a = ap.parse_args()
    z = np.load(a.src_lm); slm = z["lm"]; mu = z["mu"]; ks = list(range(a.k0, a.k1 + 1, a.step)); cols = []
    for spec in a.final:
        lab, v = spec.split("=", 1); fr = frames(v, ks); cols.append((lab, {k: fr.get(k) for k in ks}, "src"))
    for spec, lmp in zip(a.layer, a.layer_lm):
        lab, v = spec.split("=", 1); ms = {k: int(round(mu[k])) for k in ks}; fr = frames(v, list(ms.values())); llm = np.load(lmp)["lm"]
        cols.append((lab, {k: (fr.get(ms[k]), llm[ms[k]] if ms[k] < len(llm) else None, ms[k]) for k in ks}, "layer"))
    rows = []
    for k in ks:
        row = []
        for lab, d, kind in cols:
            if kind == "src":
                im = d[k]; p = slm[k]; tag = f"{lab} {k / 29.97:.2f}s"
            else:
                im, p, m = d[k]; tag = f"{lab} m{m}"
            if im is None or p is None or not np.isfinite(p[[61, 291, 13, 14], :2]).all(): row.append(np.zeros((180, 240, 3), np.uint8)); continue
            if kind == "src" and im.shape[0] != 1920 and slm.shape[0]: pass
            c = p[[61, 291, 13, 14], :2].mean(0); mw = np.linalg.norm(p[61, :2] - p[291, :2]); h = int(0.8 * mw)
            y0, x0 = int(max(0, c[1] - h)), int(max(0, c[0] - 1.3 * h)); crop = im[y0:y0 + 2 * h, x0:x0 + int(2.6 * h)]
            t_ = cv2.resize(crop, (240, 180)); cv2.putText(t_, tag, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1); row.append(t_)
        rows.append(np.hstack(row))
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); print("ok", a.out, len(ks), "frames")


if __name__ == "__main__":
    main()
