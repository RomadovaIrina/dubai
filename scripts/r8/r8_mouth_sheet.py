#!/usr/bin/env python3
"""r8 F: frame-by-frame mouth contact sheet of several videos on the SAME source timeline (rows = frames, columns = videos), crop centred on
the mouth of the first video's landmarks (MediaPipe npz), with the time stamp.
    python scripts/r8/r8_mouth_sheet.py --lm SRC.npz --videos src:SRC.mp4 m11:OUT.mp4 --t0 0 --t1 4 --out sheet.jpg [--cols-per-row 6]
"""
import argparse, numpy as np, cv2

ap = argparse.ArgumentParser(); ap.add_argument("--lm", required=True); ap.add_argument("--videos", nargs="+", required=True)
ap.add_argument("--t0", type=float, default=0); ap.add_argument("--t1", type=float, default=4); ap.add_argument("--out", required=True)
ap.add_argument("--size", type=int, default=128); ap.add_argument("--frames-per-row", type=int, default=6); ap.add_argument("--step", type=int, default=1)
a = ap.parse_args(); lm = np.load(a.lm)["lm"]
caps = [(v.split(":", 1)[0], cv2.VideoCapture(v.split(":", 1)[1])) for v in a.videos]; fps = caps[0][1].get(cv2.CAP_PROP_FPS)
k0, k1 = int(a.t0 * fps), int(a.t1 * fps); tiles = []; last = None
for lab, c in caps: c.set(cv2.CAP_PROP_POS_FRAMES, k0)
for k in range(k0, k1):
    fr = [c.read()[1] for _, c in caps]
    if (k - k0) % a.step: continue
    p = lm[min(k, len(lm) - 1)]
    if np.isfinite(p[[13, 14, 33, 263]]).all():
        cx, cy = p[[61, 291, 13, 14], 0].mean(), p[[61, 291, 13, 14], 1].mean(); r = max(16, 0.75 * np.linalg.norm(p[33, :2] - p[263, :2])); last = (cx, cy, r)
    if last is None: continue
    cx, cy, r = last; row = []
    for (lab, _), f in zip(caps, fr):
        if f is None: row.append(np.zeros((a.size, a.size, 3), np.uint8)); continue
        y0, y1, x0, x1 = int(cy - r), int(cy + r), int(cx - r), int(cx + r)
        cr = cv2.copyMakeBorder(f, 200, 200, 200, 200, cv2.BORDER_CONSTANT)[y0 + 200:y1 + 200, x0 + 200:x1 + 200]
        t = cv2.resize(cr, (a.size, a.size), interpolation=cv2.INTER_CUBIC); cv2.putText(t, f"{lab} {k / fps:.2f}", (2, 12), 0, 0.38, (0, 255, 255), 1); row.append(t)
    tiles.append(np.hstack(row))
per = a.frames_per_row; rows = [np.hstack(tiles[i:i + per] + [np.zeros_like(tiles[0])] * (per - len(tiles[i:i + per]))) for i in range(0, len(tiles), per)]
cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); print("== sheet", a.out, len(tiles), "frames")
