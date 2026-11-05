#!/usr/bin/env python3
"""r15 K: pixel-level NOSE shimmer of finals vs the source (landmarks cannot judge the nose: mediapipe's nose point moves with the mouth shape).
Inside the source nose hull (per frame, source landmarks): temporal second difference |g_t - (g_t-1 + g_t+1) / 2| per pixel (grey), mean over the hull,
for the source and each final; reported = final - source (the added shimmer), mean and p95 over frames; and the same for a cheek patch as a control.
    python k_noseflick.py --src SRC.mp4 --lm SRC_lm.npz --videos A.mp4 B.mp4 --labels a b"""
import argparse, json
import numpy as np, cv2

NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 129, 358, 64, 294, 98, 327, 48, 278, 115, 344]
CHEEK = [117, 118, 101, 36, 205, 187, 123]


def frames(p):
    c = cv2.VideoCapture(p)
    while True:
        ok, f = c.read()
        if not ok: return
        yield cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)


ap = argparse.ArgumentParser(); ap.add_argument("--src"); ap.add_argument("--lm"); ap.add_argument("--videos", nargs="+"); ap.add_argument("--labels", nargs="+")
a = ap.parse_args(); lm = np.load(a.lm)["lm"][..., :2].astype(np.float32)
gens = [frames(a.src)] + [frames(v) for v in a.videos]; buf = []; R = {lb: {"nose": [], "cheek": []} for lb in ["src"] + a.labels}; k = 0
for fs in zip(*gens):
    buf.append(fs); buf = buf[-3:]
    if len(buf) == 3 and k - 1 < len(lm) and np.isfinite(lm[k - 1]).all():
        p = lm[k - 1]; H, W = fs[0].shape
        ms = {}
        for nm, idx in (("nose", NOSE), ("cheek", CHEEK)):
            mk = np.zeros((H, W), np.uint8); cv2.fillPoly(mk, [np.round(cv2.convexHull(p[idx]))[:, 0].astype(np.int32)], 1); ms[nm] = mk > 0
        for j, lb in enumerate(["src"] + a.labels):
            d = np.abs(buf[1][j] - (buf[0][j] + buf[2][j]) / 2)
            for nm, mk in ms.items(): R[lb][nm].append(float(d[mk].mean()))
    k += 1
out = {}
for lb in a.labels:
    out[lb] = {}
    for nm in ("nose", "cheek"):
        x = np.array(R[lb][nm]) - np.array(R["src"][nm])
        out[lb][nm] = {"added_mean": round(float(x.mean()), 3), "added_p95": round(float(np.percentile(x, 95)), 3), "abs_mean": round(float(np.mean(R[lb][nm])), 3)}
out["src"] = {nm: round(float(np.mean(R["src"][nm])), 3) for nm in ("nose", "cheek")}
print(json.dumps(out))
