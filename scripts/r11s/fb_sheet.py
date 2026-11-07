#!/usr/bin/env python3
"""r11 SYNC: contact sheet of the fallback-flagged master frames: source master mouth | layer mouth (+ feature name), to see what the flags hit.
    fb_sheet.py --work W --ls L.mp4 --fb FB.npz --out sheet.jpg [--feat wash] [--max 40]"""
import argparse, numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("--work"); ap.add_argument("--ls"); ap.add_argument("--fb"); ap.add_argument("--out"); ap.add_argument("--feat", default="")
ap.add_argument("--max", type=int, default=40); ap.add_argument("--thr", type=float, default=0.5); a = ap.parse_args()
z = np.load(a.fb); lm = np.load(a.work + "/video/master_mediapipe.npz")["lm"]
feats = [k[2:] for k in z.files if k.startswith("f_")]
sel = [k for k in range(len(z["score"])) if z["score"][k] > a.thr and (not a.feat or z["f_" + a.feat][k] > a.thr)][:a.max]
def frames(p):
    c = cv2.VideoCapture(p); out = {}; k = 0
    while True:
        ok, f = c.read()
        if not ok: break
        if k in sel: out[k] = f
        k += 1
    return out
S, L = frames(a.work + "/video/master_25fps.mp4"), frames(a.ls); tiles = []
for k in sel:
    p = lm[k]; iod = np.linalg.norm(p[33, :2] - p[263, :2]); cx, cy = p[[61, 291, 13, 14], :2].mean(0); r = int(0.7 * iod)
    cr = lambda f: cv2.resize(f[max(0, int(cy - r)):int(cy + r), max(0, int(cx - r)):int(cx + r)], (160, 160))
    t = np.concatenate([cr(S[k]), cr(L[k])], 1); top = max(feats, key=lambda f_: z["f_" + f_][k])
    cv2.putText(t, f"{k} {k/25:.2f}s {top}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1); tiles.append(t)
while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(a.out, np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0), [cv2.IMWRITE_JPEG_QUALITY, 85]); print("==", len(sel), "frames ->", a.out)
