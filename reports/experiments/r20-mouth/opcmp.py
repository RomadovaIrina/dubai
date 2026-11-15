import numpy as np, glob
D = "/r18/data/jobs/"
def op(j):
    f = (glob.glob(D + j + "/e2e/work/ks/post_f0/raw_lm.npz") + glob.glob(D + j + "/e2e/work/shots/s0/ks/post_f0/raw_lm.npz"))[0]
    p = np.load(f)["lm"][..., :2]; return np.linalg.norm(p[:, 13] - p[:, 14], axis=1) / np.maximum(np.linalg.norm(p[:, 61] - p[:, 291], axis=1), 1)
for v, N in (("m03", 1013), ("b04", 1348), ("b01", 244), ("old03", 2258)):
    n, f, g = op(v + "n")[:N], op(v + "F")[:N], op(v + "G")[:N]; ok = np.isfinite(n) & np.isfinite(f) & np.isfinite(g); w = ok & (n > 0.12)
    print("%-6s opening mean now %.4f F %.4f G %.4f | corr with now F %.3f G %.3f | on wide-open frames of now (n=%d): F %.3f G %.3f of now %.3f" % (v, n[ok].mean(), f[ok].mean(), g[ok].mean(), np.corrcoef(n[ok], f[ok])[0, 1], np.corrcoef(n[ok], g[ok])[0, 1], w.sum(), f[w].mean(), g[w].mean(), n[w].mean()))
