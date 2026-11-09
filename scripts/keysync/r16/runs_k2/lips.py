import sys, numpy as np
for f in sys.argv[1:]:
    L = np.load(f)["lm"][..., :2]; w = np.linalg.norm(L[:, 61] - L[:, 291], axis=1)
    up = np.linalg.norm(L[:, 0] - L[:, 13], axis=1) / w; lo = np.linalg.norm(L[:, 14] - L[:, 17], axis=1) / w; gap = np.linalg.norm(L[:, 13] - L[:, 14], axis=1) / w
    sp = gap > 0.08   # speaking-open frames
    for nm, (k0, k1) in (("9.5-10.5s", (237, 263)), ("whole", (0, len(L)))):
        s = slice(k0, k1); m = sp[s]
        print(f"{f:10s} {nm:10s} upper lip {np.nanmean(up[s]):.3f} lower lip {np.nanmean(lo[s]):.3f} | on open frames: upper {np.nanmean(up[s][m]):.3f} lower {np.nanmean(lo[s][m]):.3f} (n {m.sum()}) | width/IOD {np.nanmean(w[s] / np.linalg.norm(L[s, 33] - L[s, 263], axis=1)):.3f}")
