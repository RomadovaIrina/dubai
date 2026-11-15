import sys, glob, os, cv2, numpy as np
# gatecal.py <dump root> <crop_lm.npz> : which luma threshold makes the gate flag the SAME keyframes under floorv as luma < 0.18 does on base
root, lmf = sys.argv[1:3]; LM = np.load(lmf)["lm"][..., :2]
def dark(tag, thr):
    out = {}
    for f in glob.glob(os.path.join(root, tag, "b*_s*_f*.jpg")):
        if "_s00_" in f: continue
        k = int(os.path.basename(f).split("_f")[1][:5])
        if k >= len(LM) or not np.isfinite(LM[k]).all(): continue
        p = LM[k][[61, 291, 0, 17, 37, 267, 84, 314]]; x0, y0 = p.min(0); x1, y1 = p.max(0); w = x1 - x0; y0 -= 0.15 * w; y1 += 0.15 * w
        im = cv2.imread(f).astype(np.float32) / 255; g = 0.114 * im[..., 0] + 0.587 * im[..., 1] + 0.299 * im[..., 2]
        sl = g[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)]; out[k] = [float((sl < t).mean()) for t in thr]
    return out
T = [0.18, 0.22, 0.25, 0.28, 0.31, 0.34]; b = dark("base", T); f = dark("fv10", T); ks = sorted(set(b) & set(f))
wb = np.array([b[k][0] > 0.12 for k in ks]); print(os.path.basename(root), "keyframes", len(ks), "base wide (luma<0.18 share>0.12):", int(wb.sum()))
for i, t in enumerate(T):
    wf = np.array([f[k][i] > 0.12 for k in ks]); print("  floorv luma<%.2f: wide %3d | both %3d | only base %3d | only floorv %3d" % (t, wf.sum(), (wb & wf).sum(), (wb & ~wf).sum(), (~wb & wf).sum()))
