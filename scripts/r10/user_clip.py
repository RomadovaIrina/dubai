#!/usr/bin/env python3
"""r11 FACE user clip: LEFT | RIGHT final videos, a face crop (eyes to neck, ~2x), the window of the most chin motion in LEFT (or --start),
labelled, the given audio (demucs mix), 1x and 0.5x, each <= --max-mb.
    python user_clip.py --left A.mp4 --right B.mp4 --audio mix.m4a --lm SRC_lm.npz --frames F_fx.frames.npz --out OUT_prefix [--dur 7 --start S]"""
import argparse, subprocess, json, numpy as np, av, pathlib
ap = argparse.ArgumentParser(); ap.add_argument("--left", required=True); ap.add_argument("--right", required=True); ap.add_argument("--audio", required=True)
ap.add_argument("--lm", required=True); ap.add_argument("--frames", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--dur", type=float, default=7.0); ap.add_argument("--start", type=float, default=-1); ap.add_argument("--max-mb", type=float, default=9.5)
ap.add_argument("--mid", default=""); ap.add_argument("--ml", default="new - chin kept"); ap.add_argument("--pick", choices=["chin", "mouth"], default="chin"); ap.add_argument("--ll", default="r9 + fix (now)"); ap.add_argument("--rl", default="new - chin kept (mask5)")
a = ap.parse_args()
c = av.open(a.left); s = c.streams.video[0]; fps = float(s.average_rate); W, H = s.codec_context.width, s.codec_context.height; c.close()
lm = np.load(a.lm)["lm"]; n = len(lm)
if a.start < 0:   # the window with the most chin motion (flow wander of the left video vs the source)
    if a.pick == "mouth":   # the most mouth movement: |d opening| of the source lips (landmarks) + of the dub mouth vs the source (frames npz)
        o_ = np.linalg.norm(lm[:, 13, :2] - lm[:, 14, :2], axis=1) / np.linalg.norm(lm[:, 33, :2] - lm[:, 263, :2], axis=1)
        do = np.nan_to_num(np.load(a.frames)["dopen"][:n]); v = np.abs(np.diff(np.nan_to_num(o_ + do), prepend=0))
    else:
        d = np.load(a.frames)["dflow"][:n]; v = np.zeros(n); ok = np.isfinite(d).all(1)
        for k in range(1, n):
            if ok[k] and ok[k - 1]: v[k] = np.linalg.norm(d[k] - d[k - 1])
    L = int(a.dur * fps); a.start = int(np.argmax(np.convolve(v, np.ones(L), "valid"))) / fps
k0 = int(a.start * fps); ks = range(k0, min(n, k0 + int(a.dur * fps))); p = np.nanmedian(lm[list(ks)], 0)
iod = float(np.linalg.norm(p[33, :2] - p[263, :2])); cx = float(p[[33, 263, 152], 0].mean()); y0 = float(p[[33, 263], 1].mean() - 0.6 * iod); y1 = float(p[152, 1] + 0.8 * iod)
h = int(y1 - y0) // 2 * 2; w = int(h * 0.75) // 2 * 2; x0 = int(max(0, min(W - w, cx - w / 2))); y0 = int(max(0, min(H - h, y0)))
ph = 720 if not a.mid else 640; pw = int(w * ph / h) // 2 * 2
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
lab = lambda t: f"drawtext=fontfile={F}:text='{t}':x=10:y=10:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6"
kf0 = int(round(a.start * fps)); kf1 = kf0 + int(round(a.dur * fps)) - 1   # frame-index selection: inputs with different time bases stay frame-aligned
rep = {"start_s": round(a.start, 2), "dur_s": a.dur, "crop": [x0, y0, w, h]}
for speed, tag in ((1.0, "1x"), (0.5, "0.5x")):
    dur_out = a.dur / speed; vb = int(a.max_mb * 8 * 1000 / dur_out - (96 if speed == 1 else 0)); out = f"{a.out}_{tag}.mp4"
    flt = (f"[0:v]select=between(n\,{kf0}\,{kf1}),setpts=N/({fps}*{speed})/TB,crop={w}:{h}:{x0}:{y0},scale={pw}:{ph}:flags=lanczos,{lab(a.ll)}[l];"
           f"[1:v]select=between(n\,{kf0}\,{kf1}),setpts=N/({fps}*{speed})/TB,crop={w}:{h}:{x0}:{y0},scale={pw}:{ph}:flags=lanczos,{lab(a.rl)}[r];"
           + (f"[2:v]select=between(n\,{kf0}\,{kf1}),setpts=N/({fps}*{speed})/TB,crop={w}:{h}:{x0}:{y0},scale={pw}:{ph}:flags=lanczos,{lab(a.ml)}[m];[l][m][r]hstack=inputs=3[v]" if a.mid else "[l][r]hstack=inputs=2[v]"))
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", a.left, "-i", a.right] + (["-i", a.mid] if a.mid else [])
    if speed == 1:
        cmd += ["-ss", str(a.start), "-t", str(a.dur), "-i", a.audio, "-filter_complex", flt, "-map", "[v]", "-map", f"{3 if a.mid else 2}:a:0", "-c:a", "aac", "-b:a", "96k"]
    else:
        cmd += ["-filter_complex", flt, "-map", "[v]", "-an"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-b:v", f"{vb}k", "-maxrate", f"{int(vb * 1.4)}k", "-bufsize", f"{vb * 2}k", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True); rep[tag] = {"file": out, "MB": round(pathlib.Path(out).stat().st_size / 1e6, 2)}
print(json.dumps(rep))
