#!/usr/bin/env python3
"""r16 L: skin colour of the generated lower face vs the source (the B "pinker" check), 512 crop grid.
Region = the lower face-oval hull minus the outer-lip hull dilated 0.12 IOD (SOURCE landmarks, same grid). Lab (L 0..100, a / b signed).
    python l_colour.py --gen G.mp4 --src S.mp4 --src-lm S.npz [--out C.json]"""
import argparse, json
import numpy as np, cv2

LOW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


def lab(vid, lm):
    cap = cv2.VideoCapture(vid); out = []; t = 0
    while t < len(lm):
        ok, f = cap.read()
        if not ok: break
        p = lm[t, :, :2]; t += 1
        m = np.zeros(f.shape[:2], np.uint8); cv2.fillPoly(m, [cv2.convexHull(p[LOW].astype(np.int32))], 1)
        l = np.zeros_like(m); cv2.fillPoly(l, [cv2.convexHull(p[LIPS].astype(np.int32))], 1); k = int(0.12 * np.linalg.norm(p[263] - p[33])) | 1
        m = (m > 0) & (cv2.dilate(l, np.ones((k, k), np.uint8)) == 0)
        L = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32); L[..., 0] *= 100 / 255; L[..., 1:] -= 128; out.append(L[m].mean(0))
    return np.array(out).mean(0)


ap = argparse.ArgumentParser(); ap.add_argument("--gen", required=True); ap.add_argument("--src", required=True); ap.add_argument("--src-lm", required=True)
ap.add_argument("--out", default=""); a = ap.parse_args()
lm = np.load(a.src_lm)["lm"]; g, s = lab(a.gen, lm), lab(a.src, lm); d = g - s
r = {"gen_Lab": [round(float(x), 1) for x in g], "src_Lab": [round(float(x), 1) for x in s], "dLab": [round(float(x), 1) for x in d], "dE": round(float(np.linalg.norm(d)), 2)}
if a.out: json.dump(r, open(a.out, "w"))
print(json.dumps(r))
