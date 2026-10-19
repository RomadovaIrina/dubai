#!/usr/bin/env python3
"""Independent check of the lower-face POSITION (track B, round 2): the common slide_metric.py and composite_register.py both use DIS
flow on the nose base / cheeks, so a registered variant scores well on it partly by construction. This check uses a DIFFERENT estimator
(phase correlation of Hann-windowed gradient-magnitude patches, cv2.phaseCorrelate; no flow, no fitting) on fixed patches from the
MASTER's landmarks, output(t) vs master(t), per LatentSync frame:
  nosebase   alae / nostrils / subnasale box (+0.10 IOD)                                     -> 2-D shift
  philtrum   between the subnasale and the upper-lip top, +-0.18 IOD around the midline       -> x shift (left-right of the mouth centre)
  mouth      outer-lip box (+0.10 IOD), x only (the mouth's left-right position; its shape differs with the dub, y is articulation)
Reported like slide_metric: abs = |d_t|, slide = |d_t - d_(t-1)| (consecutive LS frames), px at master resolution.

    python scripts/quality/r2_position_check.py --video WIN/V.mp4 --master WORK/master_25fps.mp4 --npz WORK/master_mediapipe.npz \
        --offset A --frames A:B --manifest WIN/full_index.manifest.json --out V.pos.json
"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np

NOSEBASE = [48, 278, 64, 294, 98, 327, 2, 97, 326, 164, 165, 391, 167, 393, 240, 460, 75, 305]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


def read_range(p, a, b):
    import cv2
    cap = cv2.VideoCapture(str(p)); cap.set(cv2.CAP_PROP_POS_FRAMES, a); out = []
    for _ in range(b - a):
        ok, f = cap.read()
        if not ok: break
        out.append(f)
    cap.release(); return out


def grad(g):
    import cv2
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 0.8)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def shift(m, o, box):
    import cv2
    x0, y0, x1, y1 = box
    if x1 - x0 < 8 or y1 - y0 < 8: return None
    a, b = grad(m[y0:y1, x0:x1]), grad(o[y0:y1, x0:x1]); win = cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return (dx, dy, resp)


def stats(x):
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    return {"mean": round(float(x.mean()), 4), "p95": round(float(np.percentile(x, 95)), 4), "n": int(len(x))} if len(x) else None


def main():
    import cv2
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--master", required=True); ap.add_argument("--npz", required=True)
    ap.add_argument("--offset", type=int, default=0); ap.add_argument("--frames", required=True); ap.add_argument("--manifest", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lm = np.load(a.npz)["lm"].astype(np.float64); T = len(lm); A, B = (int(x) for x in a.frames.split(":"))
    ls = np.ones(T, bool)
    if a.manifest:
        ls[:] = False
        for sg in json.loads(pathlib.Path(a.manifest).read_text())["lipsync"]["segments"]:
            if sg["action"] == "LATENT_SYNC": ls[sg["start_frame"]:sg["end_frame"]] = True
    vid = read_range(a.video, A - a.offset, B - a.offset); ref = read_range(a.master, A, A + len(vid)); B = A + len(vid)
    H, W = ref[0].shape[:2]; d = {k: np.full((B - A, 2), np.nan) for k in ("nosebase", "philtrum", "mouth")}
    for k in range(B - A):
        t = A + k; L = lm[t]
        if not ls[t] or not np.isfinite(L[0, 0]): continue
        iod = float(np.linalg.norm(L[33] - L[263])); g0, g1 = cv2.cvtColor(ref[k], cv2.COLOR_BGR2GRAY), cv2.cvtColor(vid[k], cv2.COLOR_BGR2GRAY)
        p = 0.10 * iod; nb = L[NOSEBASE]
        boxes = {"nosebase": (nb[:, 0].min() - p, nb[:, 1].min() - p, nb[:, 0].max() + p, nb[:, 1].max() + p),
                 "philtrum": (L[0, 0] - 0.18 * iod, L[2, 1], L[0, 0] + 0.18 * iod, L[0, 1]),
                 "mouth": (L[LIPS_OUTER, 0].min() - p, L[LIPS_OUTER, 1].min() - p, L[LIPS_OUTER, 0].max() + p, L[LIPS_OUTER, 1].max() + p)}
        for r, bx in boxes.items():
            bx = (int(max(0, bx[0])), int(max(0, bx[1])), int(min(W, bx[2])), int(min(H, bx[3])))
            s = shift(g0, g1, bx)
            if s is None or s[2] < 0.05: continue
            d[r][k] = s[:2]
    res = {"video": a.video, "frames": [A, B], "estimator": "phaseCorrelate on Hann-windowed gradient magnitude (no flow)", "regions": {}}
    for r, x in d.items():
        dd = np.full_like(x, np.nan); dd[1:] = x[1:] - x[:-1]
        if r == "nosebase": res["regions"][r] = {"abs_px": stats(np.linalg.norm(x, axis=1)), "slide_px": stats(np.linalg.norm(dd, axis=1)), "slide_x_px": stats(np.abs(dd[:, 0]))}
        else: res["regions"][r] = {"abs_x_px": stats(np.abs(x[:, 0])), "slide_x_px": stats(np.abs(dd[:, 0]))}
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    rr = res["regions"]; print(f"pos {pathlib.Path(a.video).name}: nosebase slide {rr['nosebase']['slide_px']} | philtrum slide_x {rr['philtrum']['slide_x_px']} | mouth slide_x {rr['mouth']['slide_x_px']}")


if __name__ == "__main__":
    main()
