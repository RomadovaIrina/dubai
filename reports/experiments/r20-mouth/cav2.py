import sys, numpy as np, cv2
INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
def stats(name, v, lmf, n):
    LM = np.load(lmf)["lm"][..., :2]; c = cv2.VideoCapture(v); R = []
    for k in range(min(n, len(LM))):
        ok, f = c.read()
        if not ok: break
        p = LM[k]
        if not np.isfinite(p).all(): continue
        mw = np.linalg.norm(p[61] - p[291]); op = np.linalg.norm(p[13] - p[14]) / max(mw, 1)
        if op < 0.12: continue
        m = np.zeros(f.shape[:2], np.uint8); cv2.fillPoly(m, [cv2.convexHull(p[INNER].astype(np.float32))[:, 0].astype(np.int32)], 1); m = cv2.erode(m, np.ones((5, 5), np.uint8))
        if m.sum() < 40: continue
        L = cv2.cvtColor(f.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[..., 0][m > 0]
        R.append((k, op, np.percentile(L, 10), np.percentile(L, 50), np.percentile(L, 90), (L < 6).mean(), L.std(), m.sum()))
    R = np.array(R); bad = R[R[:, 5] > 0.5][:, 0].astype(int)
    print("CAV %-8s open frames %4d | open med %.3f | L p10 %.1f p50 %.1f p90 %.1f | share L<6 %.3f | frames > 50%% black: %3d | %s" % (
        name, len(R), np.median(R[:, 1]), np.median(R[:, 2]), np.median(R[:, 3]), np.median(R[:, 4]), R[:, 5].mean(), len(bad), ",".join(str(round(b / 25, 1)) for b in bad[::max(1, len(bad) // 12)])), flush=True)
n = int(sys.argv[1])
for a in sys.argv[2:]:
    name, v, lm = a.split("=")[0], a.split("=")[1].split(":")[0], a.split(":")[1]; stats(name, v, lm, n)
