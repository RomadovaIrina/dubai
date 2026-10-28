#!/usr/bin/env python3
"""r7 F: mask-boundary SEAM / BLUR metric (user, 03 ~7 s: 'blurred / doubled lower face with a visible seam on the cheek and jaw' during head motion).

No manifest needed: per source frame k the pasted region C_k = { blur(|out - src| grey) > --chg } (dilated 3 px) IS the composite's footprint.
  seam_k   on the boundary band B_k (dilate(C) minus erode(C), +-r px): mean(|grad out| - |grad src|) in grey levels / px: a hard edge the source
           does not have; the feather is right when it is ~0 (the metric is signed: negative = the paste is softer than the source there).
  step_k   mean |lowpass(out) - lowpass(src)| (sigma 3 px) on B_k: a brightness / colour step across the feather (the source must continue
           smoothly into the paste).
  blur_k   inside C_k outside the lips hull (source landmarks, dilated 0.3 mouth width): mean |Laplacian out| / mean |Laplacian src|;
           < 1 = the regenerated skin is softer (blurred / doubled) than the source's.
  motion_k head speed from the source landmarks (rigid points), iod / s: the user's defect is on fast motion, so the report also gives the
           metrics on the frames with motion > --fast (default 0.3 iod/s = the p95 of 03's head motion) vs the still ones.
Output: <out>.json (series, summary, worst frames by seam and by blur), a '== seam <label> <id> ...' line; --sheet: top-N (src | out | |grad| band).
    python scripts/r7/r7_seam.py --src SRC --out-video OUT --lm <src mediapipe npz> --out x.json [--sheet x.jpg]
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np

LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]; RIGID = [33, 133, 362, 263, 168, 6, 197, 4]


def frames(path, grey=True):
    import av
    c = av.open(path)
    for fr in c.decode(c.streams.video[0]):
        yield fr.to_ndarray(format="gray" if grey else "bgr24")
    c.close()


def main() -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True); ap.add_argument("--out-video", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=""); ap.add_argument("--id", default=""); ap.add_argument("--chg", type=float, default=6.0); ap.add_argument("--band", type=int, default=4)
    ap.add_argument("--fast", type=float, default=0.3); ap.add_argument("--fps", type=float, default=30.0); ap.add_argument("--sheet", default=None); ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    lm = np.load(a.lm)["lm"]; n = len(lm); seam = np.full(n, np.nan); step = np.full(n, np.nan); blur = np.full(n, np.nan); area = np.zeros(n); mot = np.full(n, np.nan)
    kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.band + 1, 2 * a.band + 1)); k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    prev_r = None
    for k, (s, o) in enumerate(zip(frames(a.src), frames(a.out_video))):
        if k >= n: break
        p = lm[k]
        if np.isfinite(p[RIGID]).all():
            iod = float(np.linalg.norm(p[33, :2] - p[263, :2]))
            if prev_r is not None and np.isfinite(prev_r).all(): mot[k] = float(np.linalg.norm(p[RIGID, :2] - prev_r, axis=1).mean()) / max(iod, 1) * a.fps
            prev_r = p[RIGID, :2].copy()
        else: prev_r = None
        sf, of = s.astype(np.float32), o.astype(np.float32)
        C = cv2.GaussianBlur(np.abs(of - sf), (0, 0), 2) > a.chg; C = cv2.dilate(C.astype(np.uint8), k3) > 0; area[k] = int(C.sum())
        if area[k] < 100: continue
        B = (cv2.dilate(C.astype(np.uint8), kb) > 0) & ~(cv2.erode(C.astype(np.uint8), kb) > 0)
        gs = cv2.magnitude(cv2.Sobel(sf, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(sf, cv2.CV_32F, 0, 1, ksize=3)) / 8
        go = cv2.magnitude(cv2.Sobel(of, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(of, cv2.CV_32F, 0, 1, ksize=3)) / 8
        seam[k] = float((go[B] - gs[B]).mean()); step[k] = float(np.abs(cv2.GaussianBlur(of, (0, 0), 3) - cv2.GaussianBlur(sf, (0, 0), 3))[B].mean())
        inner = C.copy()
        if np.isfinite(p[LIPS_OUTER]).all():
            q = p[LIPS_OUTER, :2]; mw = float(q[:, 0].max() - q[:, 0].min()); hull = cv2.convexHull(q.astype(np.float32))[:, 0]
            lipm = np.zeros(C.shape, np.uint8); cv2.fillPoly(lipm, [np.round(hull).astype(np.int32)], 1); r = int(max(2, 0.3 * mw))
            inner &= ~(cv2.dilate(lipm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0)
        if inner.sum() > 100:
            ls_, lo_ = np.abs(cv2.Laplacian(sf, cv2.CV_32F, ksize=3))[inner].mean(), np.abs(cv2.Laplacian(of, cv2.CV_32F, ksize=3))[inner].mean()
            blur[k] = float(lo_ / max(ls_, 1e-3))
    ok = np.isfinite(seam); fast = ok & (mot > a.fast); still = ok & ~(mot > a.fast)
    def summ(m):
        return {"frames": int(m.sum()), "seam_mean": round(float(np.nanmean(seam[m])), 2) if m.any() else None, "seam_p95": round(float(np.nanpercentile(seam[m], 95)), 2) if m.any() else None,
                "step_mean": round(float(np.nanmean(step[m])), 2) if m.any() else None, "step_p95": round(float(np.nanpercentile(step[m], 95)), 2) if m.any() else None,
                "blur_mean": round(float(np.nanmean(blur[m])), 3) if m.any() else None, "blur_p05": round(float(np.nanpercentile(blur[m], 5)), 3) if m.any() else None}
    def worst(v, sign, top):
        idx = [int(i) for i in np.argsort(np.where(np.isfinite(v), sign * v, np.inf))[:top] if np.isfinite(v[i])]
        return [{"frame": i, "t_s": round(i / a.fps, 2), "seam": round(float(seam[i]), 2), "step": round(float(step[i]), 2), "blur": round(float(blur[i]), 3) if np.isfinite(blur[i]) else None, "motion_iod_s": round(float(mot[i]), 2) if np.isfinite(mot[i]) else None} for i in idx]
    rep = {"id": a.id, "label": a.label, "src": a.src, "out": a.out_video, "frames": n, "pasted_frames": int(ok.sum()), "all": summ(ok), "fast": summ(fast), "still": summ(still),
           "worst_seam": worst(seam, -1, a.top), "worst_blur": worst(blur, 1, a.top), "worst_step": worst(step, -1, a.top),
           "series": {"seam": [None if not np.isfinite(x) else round(float(x), 2) for x in seam], "step": [None if not np.isfinite(x) else round(float(x), 2) for x in step],
                      "blur": [None if not np.isfinite(x) else round(float(x), 3) for x in blur], "motion": [None if not np.isfinite(x) else round(float(x), 2) for x in mot], "area": area.astype(int).tolist()}}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.out).write_text(json.dumps(rep))
    A, Fm = rep["all"], rep["fast"]
    print(f"== seam {a.label} {a.id} pasted {rep['pasted_frames']}/{n}: seam mean {A['seam_mean']} p95 {A['seam_p95']} | step mean {A['step_mean']} p95 {A['step_p95']} | blur mean {A['blur_mean']} p05 {A['blur_p05']} || fast ({Fm['frames']} fr): seam {Fm['seam_mean']} step {Fm['step_mean']} blur {Fm['blur_mean']}")
    if a.sheet:
        ks = sorted({w["frame"] for w in rep["worst_seam"][: a.top // 2] + rep["worst_blur"][: a.top // 2]}); S = {}; O = {}
        for path, st in ((a.src, S), (a.out_video, O)):
            for k, f in enumerate(frames(path, grey=False)):
                if k in ks: st[k] = f
                if k >= max(ks): break
        tiles = []
        for k in ks:
            p = lm[k]
            if np.isfinite(p[[33, 263, 2, 152]]).all():
                iod = float(np.linalg.norm(p[33, :2] - p[263, :2])); cx = float(p[[61, 291], 0].mean()); H, W = S[k].shape[:2]
                x0, x1 = int(max(0, cx - 1.1 * iod)), int(min(W, cx + 1.1 * iod)); y0, y1 = int(max(0, p[2, 1] - 0.4 * iod)), int(min(H, p[152, 1] + 0.4 * iod))
            else: H, W = S[k].shape[:2]; x0, y0, x1, y1 = 0, 0, W, H
            sc_, oc_ = S[k][y0:y1, x0:x1], O[k][y0:y1, x0:x1]; g = cv2.cvtColor(oc_, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gm = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) / 8
            gm = cv2.applyColorMap(np.clip(gm * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            row = np.hstack([sc_, oc_, gm]); sc = 260.0 / max(row.shape[0], 1); row = cv2.resize(row, None, fx=sc, fy=sc)
            cv2.putText(row, f"{k} {k / a.fps:.2f}s seam {seam[k]:.1f} step {step[k]:.1f} blur {blur[k]:.2f} mot {mot[k]:.1f}", (4, 18), 0, 0.5, (0, 0, 255), 1); tiles.append(row)
        w_ = max(t.shape[1] for t in tiles); tiles = [np.pad(t, ((0, 2), (0, w_ - t.shape[1]), (0, 0)), constant_values=255) for t in tiles]
        cv2.imwrite(a.sheet, np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return 0


if __name__ == "__main__":
    sys.exit(main())
