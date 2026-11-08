#!/usr/bin/env python3
"""KeySync bake-off prep: a static square face crop of the source at 25 fps 512x512 (KeySync's input), the dub speech at 16 kHz, box.json for the paste-back."""
import json, subprocess, sys, numpy as np
vid, lmf, wav, out, tag = sys.argv[1:6]
lm = np.load(lmf)["lm"]                     # (T, 478, 2) source pixels
p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", vid], capture_output=True, text=True).stdout.strip().split(",")
W, H = int(p[0]), int(p[1])
ok = np.isfinite(lm).all(axis=(1, 2)) & (lm.max(axis=(1, 2)) > 0)
mn, mx = lm[ok].min(axis=1), lm[ok].max(axis=1)
cx, cy = np.median((mn[:, 0] + mx[:, 0]) / 2), np.median((mn[:, 1] + mx[:, 1]) / 2)
fh = np.percentile(mx[:, 1] - mn[:, 1], 90); mov = max(np.ptp((mn[:, 0] + mx[:, 0]) / 2), np.ptp((mn[:, 1] + mx[:, 1]) / 2))
side = int(min(W, H, float(__import__("os").environ.get("KS_BOX_K", 1.7)) * fh + float(__import__("os").environ.get("KS_BOX_M", 1)) * mov)) // 2 * 2
x0 = int(np.clip(cx - side / 2, 0, W - side)); y0 = int(np.clip(cy - side / 2 + 0.05 * side, 0, H - side))
x0, y0 = x0 // 2 * 2, y0 // 2 * 2   # r16b K1 (audit P2): even origin = what ffmpeg's yuv420p crop really does, so box.json and the crop agree
json.dump({"x": x0, "y": y0, "side": side, "W": W, "H": H, "face_h": float(fh), "move": float(mov)}, open(f"{out}/box_{tag}.json", "w"))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vid, "-an", "-vf", f"crop={side}:{side}:{x0}:{y0},fps=25,scale=512:512:flags=lanczos", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", f"{out}/videos/{tag}.mp4"], check=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav, "-ac", "1", "-ar", "16000", f"{out}/audios/{tag}.wav"], check=True)
print(tag, json.load(open(f"{out}/box_{tag}.json")))
