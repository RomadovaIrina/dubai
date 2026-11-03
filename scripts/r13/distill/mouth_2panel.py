#!/usr/bin/env python3
"""r13 distillation pilot: a mouth 2-panel clip (left | right), the same crop of both videos around the median mouth of the source landmarks
(MediaPipe 478, the composite's source_lm npz), 2.6 x the mouth width, scaled to 360 px high each, labels, the left video's audio.
    python scripts/r13/distill/mouth_2panel.py --left k5.mp4 --right student.mp4 --lm source_lm_F.npz --out clip.mp4 [--labels "k5 (production)" "student 4 steps"]"""
import argparse, subprocess
import numpy as np

MOUTH = [61, 291, 0, 17, 13, 14, 78, 308]   # MediaPipe lip corners / centre points


def probe(v):
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", v], capture_output=True, text=True).stdout.strip()
    w, h = o.split(",")[:2]; return int(w), int(h)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--left", required=True); ap.add_argument("--right", required=True); ap.add_argument("--lm", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--labels", nargs=2, default=["k5 (production)", "student 4 steps"]); a = ap.parse_args()
    z = np.load(a.lm); key = next(k for k in ("lm", "lm_all", "landmarks") if k in z.files); L = z[key]
    if L.ndim == 4: L = L[:, 0]
    W, H = probe(a.left); P = L[:, MOUTH, :2].reshape(-1, 2); P = P[np.isfinite(P).all(1)]
    if np.nanmax(P) <= 1.5: P = P * np.array([W, H])
    cx, cy = np.median(P, 0); mw = np.percentile(P[:, 0], 95) - np.percentile(P[:, 0], 5); s = int(max(64, 2.6 * mw)); s -= s % 2
    x0 = int(np.clip(cx - s / 2, 0, W - s)); y0 = int(np.clip(cy - s / 2, 0, H - s))
    lab = lambda t: f"drawtext=text='{t}':x=8:y=8:fontsize=20:fontcolor=yellow:box=1:boxcolor=black@0.6"
    fc = (f"[0:v]crop={s}:{s}:{x0}:{y0},scale=-2:360,{lab(a.labels[0])}[l];[1:v]crop={s}:{s}:{x0}:{y0},scale=-2:360,{lab(a.labels[1])}[r];[l][r]hstack=inputs=2[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.left, "-i", a.right, "-filter_complex", fc, "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-crf", "18",
                    "-preset", "fast", "-c:a", "aac", "-b:a", "128k", "-shortest", a.out], check=True)
    print(f"crop {s}x{s} at ({x0},{y0}) -> {a.out}")


if __name__ == "__main__":
    main()
