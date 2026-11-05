#!/usr/bin/env python3
"""r6 m03 'explosion' hunt: per-frame mouth pop (explode.pop_at: |f_k - mean of the flow-warped neighbours| in the lip hull + 0.3 IOD)
and mouth luma / gloss (share of near-white pixels in the lip hull) of several finals on the SOURCE timeline (source landmarks), in a
time window; prints the top frames per final and writes a contact sheet (crops at the top frames of the first final, every final + src).
    python boom_scan.py --src SRC --src-lm source_lm_F.npz --t0 6 --t1 11 --final a=A.mp4 --final b=B.mp4 ... --sheet out.jpg"""
import argparse, sys, json
import numpy as np, cv2
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import explode as X


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--src-lm", required=True)
    ap.add_argument("--t0", type=float, default=6); ap.add_argument("--t1", type=float, default=11); ap.add_argument("--final", action="append", required=True)
    ap.add_argument("--sheet"); ap.add_argument("--json"); a = ap.parse_args()
    z = np.load(a.src_lm); slm = z["lm"].astype(np.float64)
    cap = cv2.VideoCapture(a.src); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    S = X.read_all(a.src); k0, k1 = int(a.t0 * fps), min(int(a.t1 * fps), len(S) - 2, len(slm) - 2)
    ps = np.array([X.pop_at(S, slm, k) for k in range(k0, k1)])
    res = {"fps": fps, "k0": k0, "src_pop": ps.tolist()}; cols = {}
    def gloss(F, k):
        p = slm[k]; q = p[X.LIPS, :2]; m = np.zeros(F[k].shape[:2], np.uint8); cv2.fillPoly(m, [np.round(cv2.convexHull(q.astype(np.float32))[:, 0]).astype(np.int32)], 1)
        v = F[k][m > 0]; return float(v.mean()), float((v > 200).mean())
    for spec in a.final:
        lab, v = spec.split("=", 1); Fv = X.read_all(v)
        if Fv[0].shape != S[0].shape: Fv = [cv2.resize(x, (S[0].shape[1], S[0].shape[0])) for x in Fv]
        pv = np.array([X.pop_at(Fv, slm, k) for k in range(k0, k1)]); ex = pv - ps
        gl = np.array([gloss(Fv, k) for k in range(k0, k1)])
        top = np.argsort(-np.nan_to_num(ex))[:6]
        print(f"{lab}: pop excess max {np.nanmax(ex):.2f} at t={(k0 + int(np.nanargmax(ex))) / fps:.2f} s; top " +
              ", ".join(f"{(k0 + i) / fps:.2f}s:{ex[i]:.1f}" for i in top) + f"; lip luma max {gl[:, 0].max():.0f} (median {np.median(gl[:, 0]):.0f}),"
              f" bright share max {gl[:, 1].max():.3f} at t={(k0 + int(gl[:, 1].argmax())) / fps:.2f}", flush=True)
        res[lab] = {"pop_excess": np.round(ex, 2).tolist(), "lip_luma": np.round(gl[:, 0], 1).tolist(), "bright": np.round(gl[:, 1], 4).tolist()}
        cols[lab] = (v, top)
    if a.json: open(a.json, "w").write(json.dumps(res))
    if a.sheet:
        lab0 = a.final[0].split("=", 1)[0]; ks = sorted(set(int(k0 + i) for i in cols[lab0][1][:4]))
        allc = {lab: X.read_all(v, gray=False) for lab, (v, _) in cols.items()}; Sc = X.read_all(a.src, gray=False); rows = []
        for k in ks:
            p = slm[k]; c = p[X.LIPS, :2].mean(0); mw = np.linalg.norm(p[61, :2] - p[291, :2]); h = int(0.9 * mw)
            y0, x0 = int(max(0, c[1] - h)), int(max(0, c[0] - 1.2 * h)); y1, x1 = y0 + 2 * h, x0 + int(2.4 * h); row = []
            for lab, C in list(allc.items()) + [("src", Sc)]:
                im = C[k]; im = cv2.resize(im, (Sc[0].shape[1], Sc[0].shape[0])) if im.shape != Sc[0].shape else im
                t_ = cv2.resize(im[y0:y1, x0:x1], (240, 200)); cv2.putText(t_, f"{lab} {k / fps:.2f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1); row.append(t_)
            rows.append(np.hstack(row))
        cv2.imwrite(a.sheet, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); print("sheet", a.sheet)


if __name__ == "__main__":
    main()
