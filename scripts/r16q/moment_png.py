#!/usr/bin/env python3
"""Frame strip for one moment: top row = the final, bottom row = the source, face crop (r16q_detect box, lower 2/3 of the face),
frames t-6..t+6 every 2nd frame (25 fps grid), each labelled with its time.
    python3 moment_png.py --final F --src S --npz det.npz --t 12.34 --out X.png [--label TEXT] [--step 2] [--half 6]"""
import argparse, subprocess, sys, pathlib
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "quality"))
import r16q_detect as D


def grab(v, t0, n, W, H):
    return list(D.reader(v, W, H, t0, t0 + n / D.FPS + 0.02))[:n]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--final"); ap.add_argument("--src"); ap.add_argument("--npz"); ap.add_argument("--t", type=float)
    ap.add_argument("--out"); ap.add_argument("--label", default=""); ap.add_argument("--step", type=int, default=2); ap.add_argument("--half", type=int, default=6)
    ap.add_argument("--y0", type=float, default=0.38); ap.add_argument("--y1", type=float, default=1.0); ap.add_argument("--tile", type=int, default=300); ap.add_argument("--x0", type=float, default=0.12); ap.add_argument("--x1", type=float, default=0.88)
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=True); boxes = z["boxes"]
    W, H, _ = D.probe(a.src); sc = min(1.0, 1920 / max(W, H)); W, H = int(round(W * sc / 2) * 2), int(round(H * sc / 2) * 2)
    k = int(round(a.t * D.FPS)); k0 = max(0, k - a.half); k1 = min(len(boxes) - 1, k + a.half)
    fo = grab(a.final, k0 / D.FPS, k1 - k0 + 1, W, H); fs = grab(a.src, k0 / D.FPS, k1 - k0 + 1, W, H)
    rows = [[], []]; T = a.tile
    for j in range(0, min(len(fo), len(fs)), a.step):
        kk = k0 + j; x0, y0, s = boxes[min(kk, len(boxes) - 1)]
        if not np.isfinite([x0, y0, s]).all(): continue
        x0, y0, s = int(x0), int(y0), int(s); ya = y0 + int(a.y0 * s); xa = x0 + int(a.x0 * s)
        for r, f in enumerate((fo[j], fs[j])):
            c = f[max(0, ya):max(0, ya) + int((a.y1 - a.y0) * s), max(0, xa):max(0, xa) + int((a.x1 - a.x0) * s)]
            if c.size == 0: c = np.zeros((T, T, 3), np.uint8)
            c = cv2.resize(c, (T, int(T * c.shape[0] / max(c.shape[1], 1))))
            col = (0, 0, 255) if kk == k else (255, 255, 255)
            cv2.putText(c, f"{kk / D.FPS:.2f}{' *' if kk == k else ''}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(c, f"{kk / D.FPS:.2f}{' *' if kk == k else ''}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
            rows[r].append(c)
    h = min(min(x.shape[0] for x in rr) for rr in rows)
    img = np.vstack([np.hstack([x[:h] for x in rr]) for rr in rows])
    bar = np.full((26, img.shape[1], 3), 32, np.uint8)
    cv2.putText(bar, (a.label + "   top: final  /  bottom: source")[:160], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(a.out, np.vstack([bar, img]), [cv2.IMWRITE_JPEG_QUALITY, 88] if a.out.endswith(".jpg") else [])


if __name__ == "__main__":
    main()
