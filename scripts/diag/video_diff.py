#!/usr/bin/env python3
"""Diagnostic: per-frame difference map original(25 fps master) vs first_res video: where was the face re-rendered (LatentSync+CodeFormer), and is
there any change OUTSIDE the face box (seams / crossfade bleed).  python scripts/diag/video_diff.py --id 03 --box x,y,w,h"""
import argparse, subprocess, json, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--id"); ap.add_argument("--box"); a = ap.parse_args()
src = f"/workspace/dub/test_videos/{'01.MP4' if a.id == '01' else '03.mp4'}"; dub = f"/workspace/dub/first_res/{a.id}_e2e_final.mp4"
W, H = (480, 848) if a.id == "01" else (464, 848)
def frames(f, fps):
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", f, "-vf", f"fps={fps},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    n = W * H
    while True:
        b = p.stdout.read(n)
        if len(b) < n: return
        yield np.frombuffer(b, np.uint8).reshape(H, W).astype(np.float32)
x, y, w, h = map(int, a.box.split(","))
out = []
for i, (o, d) in enumerate(zip(frames(src, 25), frames(dub, 25))):
    diff = np.abs(o - d); m = np.zeros_like(diff, bool); m[y:y + h, x:x + w] = True
    out.append((i / 25, float(diff[m].mean()), float(diff[~m].mean())))
out = np.array(out); np.save(f"/workspace/dub/tts_diag/visual/{a.id}_diff.npy", out)
inside = out[:, 1] > 6; 
segs = []; i = 0
while i < len(out):
    if inside[i]:
        j = i
        while j + 1 < len(out) and inside[j + 1]: j += 1
        if j - i >= 5: segs.append((round(out[i, 0], 2), round(out[j, 0], 2)))
        i = j + 1
    else: i += 1
print(a.id, "frames", len(out), "| face-box changed segments (>=0.2s, meanAbsDiff>6):", segs)
print("outside-box mean diff: median %.2f p99 %.2f max %.2f (at t=%.2f)" % (np.median(out[:, 2]), np.percentile(out[:, 2], 99), out[:, 2].max(), out[out[:, 2].argmax(), 0]))
print("inside-box diff: median %.2f p90 %.2f" % (np.median(out[:, 1]), np.percentile(out[:, 1], 90)))
