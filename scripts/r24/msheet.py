#!/usr/bin/env python3
# msheet.py out.jpg lm_src.npz a:b:step name=video ...   lower-face crops (from the source landmarks) of consecutive frames, one row per video
import sys, numpy as np, cv2
out, lmp, rng = sys.argv[1:4]; a, b, st = [int(x) for x in rng.split(":")]; V = [x.split("=", 1) for x in sys.argv[4:]]
L = np.load(lmp)["lm"][..., :2]; idx = list(range(a, b, st)); T = 150; rows = []
for nm, p in V:
    c = cv2.VideoCapture(p); c.set(cv2.CAP_PROP_POS_FRAMES, a); F = {}
    for k in range(a, b):
        ok, f = c.read()
        if not ok: break
        F[k] = f
    tiles = []
    for k in idx:
        q = L[k]; cx, cy = q[[61, 291, 152, 2], 0].mean(), (q[2, 1] + q[152, 1]) / 2; s = 1.25 * (q[152, 1] - q[2, 1])
        x0, y0 = int(cx - s / 2), int(cy - s / 2); f = F.get(k)
        t = np.zeros((T, T, 3), np.uint8) if f is None else cv2.resize(cv2.copyMakeBorder(f, 400, 400, 400, 400, cv2.BORDER_CONSTANT)[y0 + 400:y0 + 400 + int(s), x0 + 400:x0 + 400 + int(s)], (T, T), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"{nm} {k / 30:.2f}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1); tiles.append(t)
    rows.append(np.hstack(tiles))
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); print(out)
