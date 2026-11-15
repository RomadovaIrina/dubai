import sys, cv2, numpy as np
# tj.py <src512.mp4> <crop_lm.npz> name=final512.mp4 ... : temporal jitter of the residual (final - source) inside the lower face, luma levels per frame; by second 24..32 + overall
src, lmf = sys.argv[1:3]; LM = np.load(lmf)["lm"][..., :2]
LOW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 2]
def gray(p):
    c = cv2.VideoCapture(p); F = []
    while True:
        ok, f = c.read()
        if not ok: break
        F.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    return F
S = gray(src)
for a in sys.argv[3:]:
    name, p = a.split("=", 1); G = gray(p); n = min(len(S), len(G), len(LM)); J = np.full(n, np.nan); prev = None
    for k in range(n):
        q = LM[k]
        if not np.isfinite(q).all(): prev = None; continue
        m = np.zeros(S[k].shape, np.uint8); cv2.fillPoly(m, [cv2.convexHull(q[LOW].astype(np.float32))[:, 0].astype(np.int32)], 1); r = cv2.GaussianBlur(G[k] - S[k], (0, 0), 2)
        if prev is not None: mm = (m & prev[1]) > 0; J[k] = float(np.abs(r - prev[0])[mm].mean()) if mm.sum() > 100 else np.nan
        prev = (r, m)
    print("%-6s residual jitter: all med %.2f p95 %.2f | by second 24..32: %s" % (name, np.nanmedian(J), np.nanpercentile(J, 95), [round(float(np.nanmedian(J[t * 25:(t + 1) * 25])), 2) for t in range(24, 33)]))
