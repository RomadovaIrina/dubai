import sys, numpy as np
# ref0.py feat.npz [n]: how good is frame 0 as the reference of every 169-frame batch
d = np.load(sys.argv[1]); F = d["F"]; K = [str(k) for k in d["keys"]]; f = {k: F[:, i] for i, k in enumerate(K)}; N = len(F); V = np.isfinite(F).all(1)
out = []
for b in range(-(-N // 169)):
    B = np.flatnonzero(V[b * 169:(b + 1) * 169]) + b * 169
    if not len(B) or not V[0]: out.append(None); continue
    tone = np.array([np.median(f[c][B]) for c in ("L", "A", "B")]); dE = float(np.linalg.norm(np.array([f["L"][0], f["A"][0], f["B"][0]]) - tone))
    out.append((round(dE, 1), round(float(f["eyeL"][0] - np.median(f["eyeL"][B])), 1), round(float(abs(f["yaw"][0] - np.median(f["yaw"][B]))), 2), round(float(np.hypot(f["pL"][0] - np.median(f["pL"][B]), f["pA"][0] - np.median(f["pA"][B]))), 1)))
print(sys.argv[1].split("/")[-1], "frame 0 vs batch (dE tone, eye L diff, yaw diff, upper-lip dE):", out)
