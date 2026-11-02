#!/usr/bin/env python3
"""Mouth sheet for the gate: N frames of an open-mouth window, mouth crop (from the source landmarks), tiled in one row per video.
   ks_sheet.py --lm SRC_lm.npz --fps 30 --t0 60.7 --n 8 --step 0.2 --out sheet.png  label=video.mp4 [label=video.mp4 ...]"""
import argparse, subprocess, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--lm"); ap.add_argument("--fps", type=float, default=30); ap.add_argument("--t0", type=float)
ap.add_argument("--n", type=int, default=8); ap.add_argument("--step", type=float, default=0.2); ap.add_argument("--out"); ap.add_argument("vids", nargs="+"); a = ap.parse_args()
lm = np.load(a.lm)["lm"]; i0, i1 = int(a.t0 * a.fps), int((a.t0 + a.n * a.step) * a.fps)
m = lm[i0:i1, [61, 291, 13, 14, 0, 17]].reshape(-1, 2); cx, cy = np.median(m, 0); iod = np.median(np.linalg.norm(lm[i0:i1, 33] - lm[i0:i1, 263], axis=1))
w = int(iod * 1.3) // 2 * 2; h = int(w * 0.75) // 2 * 2; x, y = int(cx - w / 2), int(cy - h / 2)
rows = []
for k, lv in enumerate(a.vids):
    lab, v = lv.split("=", 1)
    sel = "+".join(f"eq(n\,{int(round((a.t0 + j * a.step) * a.fps))})" for j in range(a.n))
    o = f"/tmp/sheet_row{k}.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", v, "-vf", f"select='{sel}',crop={w}:{h}:{x}:{y},scale=240:-2,tile={a.n}x1,"
                    f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='{lab}':fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6:x=6:y=6",
                    "-frames:v", "1", "-vsync", "vfr", o], check=True); rows.append(o)
subprocess.run(["ffmpeg", "-v", "error", "-y"] + sum([["-i", r] for r in rows], []) + ["-filter_complex", "".join(f"[{i}:v]" for i in range(len(rows))) + f"vstack={len(rows)}" if len(rows) > 1 else "null", a.out], check=True)
print("sheet", a.out)
