#!/usr/bin/env python3
"""How much of a video KeySync could skip: 25 fps frames inside runs (>= --min-run s) where the dub is silent AND the source mouth is closed,
minus --pad s at each run end (the ramps).  ks_skip.py <dub 16k wav> <source lm npz (30 fps, 478 pts)> <src fps>"""
import sys, numpy as np
from scipy.io import wavfile
wav, lmf, sfps = sys.argv[1], sys.argv[2], float(sys.argv[3]); MIN, PAD = 1.0, 0.3
sr, a = wavfile.read(wav); a = a.astype(np.float32) / (np.abs(a).max() + 1e-9)
n = int(len(a) / sr * 25); hop = sr // 25
rms = np.array([np.sqrt(np.mean(a[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
db = 20 * np.log10(rms); silent = db < np.percentile(db, 95) - 35
lm = np.load(lmf)["lm"]; t = np.arange(n) / 25; j = np.clip(np.round(t * sfps).astype(int), 0, len(lm) - 1)
gap = np.linalg.norm(lm[j, 13] - lm[j, 14], axis=1) / np.linalg.norm(lm[j, 33] - lm[j, 263], axis=1)
closed = gap < 0.04
for name, m in (("dub silent", silent), ("silent & closed", silent & closed)):
    keep = np.zeros(n, bool); i = 0
    while i < n:
        if m[i]:
            k = i
            while k < n and m[k]: k += 1
            if (k - i) / 25 >= MIN:
                keep[i + int(PAD * 25):k - int(PAD * 25)] = True
            i = k
        else: i += 1
    print(f"{name:16s}: {m.mean() * 100:5.1f} % of frames, skippable (runs >= {MIN} s, pad {PAD} s) {keep.mean() * 100:5.1f} %")
