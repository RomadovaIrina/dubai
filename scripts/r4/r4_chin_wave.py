#!/usr/bin/env python3
"""Round 4 V task 5 gate metric: chin / jaw "wave" = the regenerated lower chin deforming slowly against the source.
Per frame: residual OUT - SOURCE (grey, Gaussian sigma 3 = low frequencies, no encoder speckle), warped to a canonical head frame (eyes / nose tip /
chin tip similarity, 300 px, IOD 100 px). Fixed canonical mask: inside the mean jaw contour, more than 0.15 IOD below the mean lower lip.
Score = per-pixel temporal std of that residual over rolling 2 s windows (frames with a face), mean over the mask; mean and worst window.
A source-identical chin scores the encoder floor (~0.5-1); 03 v1 55-72 s 2.27, chin-clip 0.5 1.61. Frames are decoded sequentially (cv2 seeks on
the raw inputs land one frame early).
    python scripts/r4/r4_chin_wave.py --video OUT.mp4 --src SRC.mp4 --lm SRC_lm_F.npz --out x.json"""
from __future__ import annotations
import argparse, json, pathlib
import cv2, numpy as np
JAW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
CANON = np.float32([[100, 90], [200, 90], [150, 150], [150, 230]])   # eyes 33 / 263, nose tip 1, chin 152


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--src", required=True); ap.add_argument("--lm", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--win-s", type=float, default=2.0); a = ap.parse_args()
    lm = np.load(a.lm)["lm"]; co, cs = cv2.VideoCapture(a.video), cv2.VideoCapture(a.src); fps = co.get(cv2.CAP_PROP_FPS) or 30.0
    Ms, jaws, lips = {}, [], []
    for k in range(len(lm)):
        p = lm[k]
        if not np.isfinite(p[[33, 263, 1, 152, 17] + JAW]).all(): continue
        M, _ = cv2.estimateAffinePartial2D(p[[33, 263, 1, 152], :2].astype(np.float32), CANON)
        if M is None: continue
        Ms[k] = M; jaws.append(cv2.transform(p[JAW, :2][None].astype(np.float32), M)[0]); lips.append(cv2.transform(p[[17], :2][None].astype(np.float32), M)[0, 0, 1])
    if not Ms: pathlib.Path(a.out).write_text(json.dumps({"video": a.video, "chin_wave": None})); return
    mask = np.zeros((300, 300), np.uint8); cv2.fillPoly(mask, [np.round(np.mean(jaws, 0)).astype(np.int32)], 1); mask[: int(np.mean(lips) + 15)] = 0
    sel = mask.astype(bool); R = np.full((len(lm), int(sel.sum())), np.nan, np.float32)
    for k in range(len(lm)):
        ok1, o = co.read(); ok2, s = cs.read()
        if not (ok1 and ok2): break
        if k not in Ms: continue
        d = cv2.GaussianBlur(cv2.cvtColor(o, cv2.COLOR_BGR2GRAY).astype(np.float32) - cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 3)
        R[k] = cv2.warpAffine(d, Ms[k], (300, 300))[sel]
    w = max(int(round(a.win_s * fps)), 2); vals, ts = [], []
    for k0 in range(0, len(lm) - w + 1, w // 2):
        blk = R[k0:k0 + w]; blk = blk[np.isfinite(blk).all(1)]
        if len(blk) >= w // 2: vals.append(float(blk.std(0).mean())); ts.append(round(k0 / fps, 1))
    res = {"video": a.video, "mask_px": int(sel.sum()), "windows": len(vals),
           "chin_wave": {"mean": round(float(np.mean(vals)), 3) if vals else None, "worst_2s": round(float(np.max(vals)), 3) if vals else None,
                         "worst_t": ts[int(np.argmax(vals))] if vals else None, "p90": round(float(np.percentile(vals, 90)), 3) if vals else None}}
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res["chin_wave"]))


if __name__ == "__main__":
    main()
