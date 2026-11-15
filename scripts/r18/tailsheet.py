import sys, numpy as np, cv2
# tailsheet.py <out.jpg> <lm_f0.npz> <raw_lm.npz> t0 t1 step_frames name=path ... : mouth crops over time (rows = files, columns = time) + openness series (source / drawn)
out, lmp, klp, t0, t1, st = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]); files = [a.split("=", 1) for a in sys.argv[7:]]
z = np.load(lmp); LM = z["lm"][..., :2].astype(np.float32); J = z["j"] if "j" in z.files else np.arange(len(LM)); KL = np.load(klp)["lm"][..., :2].astype(np.float32)
def opn(p):
    if not np.isfinite(p[[13, 14, 61, 291]]).all(): return np.nan
    return float(np.linalg.norm(p[13] - p[14]) / max(float(np.linalg.norm(p[61] - p[291])), 1e-3))
k0, k1 = int(t0 * 25), int(t1 * 25); print("lm", LM.shape, "ks lm", KL.shape, "max coord", float(np.nanmax(LM[k0])), float(np.nanmax(KL[min(k0, len(KL) - 1)])))
print("frame  t      src   drawn")
for k in range(k0, min(k1, len(LM)), 2):
    print("%5d %6.2f  %.3f  %.3f" % (k, k / 25, opn(LM[min(int(J[min(k, len(J) - 1)]), len(LM) - 1)]), opn(KL[min(k, len(KL) - 1)])))
rows = []
for name, p in files:
    c = cv2.VideoCapture(p); W, H = c.get(3), c.get(4); n = int(c.get(7)); pts = LM[k0:k1, [61, 291, 13, 14, 152, 1]].reshape(-1, 2); pts = pts[np.isfinite(pts).all(1)]
    sx = W if pts.max() <= 2 else 1.0; sy = H if pts.max() <= 2 else 1.0; cx, cy = np.median(pts[:, 0]) * sx, np.median(pts[:, 1]) * sy; r = int(0.20 * H * 0.5)
    x0, y0 = int(max(0, cx - r)), int(max(0, cy - r * 0.8)); tiles = []
    for k in range(k0, k1, st):
        c.set(cv2.CAP_PROP_POS_FRAMES, min(k, n - 1)); ok, f = c.read()
        if not ok: break
        t = cv2.resize(f[y0:y0 + int(1.6 * r), x0:x0 + 2 * r], (200, 160), interpolation=cv2.INTER_CUBIC); cv2.putText(t, "%.2f" % (k / 25), (4, 14), 0, 0.42, (255, 255, 255), 1); tiles.append(t)
    row = np.hstack(tiles); cv2.putText(row, name, (4, 152), 0, 0.5, (0, 255, 255), 1); rows.append(row)
w = min(r.shape[1] for r in rows); cv2.imwrite(out, np.vstack([r[:, :w] for r in rows]), [cv2.IMWRITE_JPEG_QUALITY, 88])
