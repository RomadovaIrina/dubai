#!/usr/bin/env python3
"""r9 diag: at which step does the output mouth follow the SOURCE mouth instead of the dub? Mouth opening (inner-lip gap / face height, MediaPipe
on each 25 fps file) correlated with the source mouth opening and with the dub energy, for: the raw LatentSync layer (pre-CodeFormer), after
CodeFormer, after the detail transfer + SR (the composite input), and the final output.   python layer_copy.py <id>"""
import pathlib, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, "/workspace/dub/dubai/scripts/pilot"); sys.path.insert(0, "/workspace/dub/dubai/scripts/r9")
import clean_face_stack as cfs, soundfile as sf
from sync_diag import opening, corr
i = sys.argv[1]; sfx = sys.argv[2] if len(sys.argv) > 2 else ""; B = pathlib.Path("/workspace/dub/artifacts/r4m"); G = f"r9g_{i}{sfx}"; W = B / G / f"work_{i}_{G}"; st = f"{i}_{G}_25fps"; t = pathlib.Path(tempfile.mkdtemp())
src = np.load(B / f"r9g_{i}" / f"work_{i}_r9g_{i}" / "video" / "master_mediapipe.npz")["lm"]; os_ = opening(src); n = len(src)
y, sr = sf.read(f"/workspace/dub/artifacts/r4a/r9_{i}/{i}/dubbed_16k.wav"); hop = sr // 25
env = np.array([np.sqrt(np.mean(y[k * hop:(k + 1) * hop] ** 2)) if k * hop < len(y) else 0 for k in range(n)])
ys, _ = sf.read(str(B / f"r9g_{i}" / f"work_{i}_r9g_{i}" / "video" / "own_audio_16k.wav")); envs = np.array([np.sqrt(np.mean(ys[k * 640:(k + 1) * 640] ** 2)) if k * 640 < len(ys) else 0 for k in range(n)])
o25 = t / "out25.mp4"; subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(B / f"r9f_{i}{sfx}" / f"{i}_r9f_{i}{sfx}.mp4"), "-r", "25", "-an", "-c:v", "libx264", "-crf", "12", str(o25)], check=True)
print(f"source mouth vs source audio {corr(os_, envs)} | source mouth vs DUB audio {corr(os_, env)}")
for name, p in [("1 LatentSync raw (pre-CodeFormer)", W / f"{st}_ls_pre.mp4"), ("2 + CodeFormer", W / f"{st}_ls_raw.mp4"), ("3 + detail transfer + SR (composite input)", W / f"{st}_ls_cfd_cfd_srt1.mp4"), ("4 final output", o25)]:
    lm = np.load(cfs.run_mediapipe(p, t / (p.stem + "_mp.npz")))["lm"][:n]; o = opening(lm)
    print(f"{name:45s} mouth vs SOURCE mouth {corr(o, os_)} | vs DUB audio {corr(o, env)} | mean opening out/src {np.nanmean(o) / np.nanmean(os_):.2f}")
