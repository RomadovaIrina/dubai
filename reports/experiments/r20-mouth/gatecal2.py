import sys, glob, os, cv2, numpy as np
# gatecal2.py <dump root> <crop_lm.npz> : the gate under floorv = black pixels OR the recorded would-be-black cells; flagged keyframes vs the base run (luma < 0.18 share > 0.12)
root, lmf = sys.argv[1:3]; LM = np.load(lmf)["lm"][..., :2]
def box(k):
    p = LM[k][[61, 291, 0, 17, 37, 267, 84, 314]]; x0, y0 = p.min(0); x1, y1 = p.max(0); w = x1 - x0; return max(0, int(x0)), max(0, int(y0 - 0.15 * w)), int(x1), int(y1 + 0.15 * w)
def dark(tag):
    out = {}; M = {}
    for f in sorted(glob.glob(os.path.join(root, tag, "fw_b*.npy"))):
        b0 = int(os.path.basename(f)[4:7]); a = np.load(f)
        for i in range(len(a)): M[(b0 + i // 14, i % 14)] = a[i]
    for f in glob.glob(os.path.join(root, tag, "b*_s*_f*.jpg")):
        if "_s00_" in f: continue
        nm = os.path.basename(f); b, s_, k = int(nm[1:4]), int(nm[6:8]), int(nm.split("_f")[1][:5])
        if k >= len(LM) or not np.isfinite(LM[k]).all(): continue
        x0, y0, x1, y1 = box(k); im = cv2.imread(f).astype(np.float32) / 255; g = 0.114 * im[..., 0] + 0.587 * im[..., 1] + 0.299 * im[..., 2]; dk = g < 0.18
        if (b, s_) in M: dk = dk | (np.repeat(np.repeat(M[(b, s_)], 8, 0), 8, 1) > 0.5)
        sl = dk[y0:y1, x0:x1]; out[k] = float(sl.mean()) if sl.size else 0.0
    return out
B = dark("cb"); kb = sorted(B); wb = np.array([B[k] > 0.12 for k in kb]); print(os.path.basename(root), "keyframes", len(kb), "release wide:", int(wb.sum()))
for t in sorted(os.path.basename(d) for d in glob.glob(os.path.join(root, "c*_*"))):
    F = dark(t); ks = [k for k in kb if k in F]; a = np.array([B[k] > 0.12 for k in ks]); f = np.array([F[k] > 0.12 for k in ks])
    print("  %-10s wide %3d | both %3d | only release %3d | only floorv %3d | mean |d dark| %.3f" % (t, f.sum(), (a & f).sum(), (a & ~f).sum(), (~a & f).sum(), float(np.mean([abs(B[k] - F[k]) for k in ks]))))
