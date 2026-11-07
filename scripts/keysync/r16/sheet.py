import sys, numpy as np, cv2
# sheet.py out.jpg lm.npz t0 t1 step vid1 vid2 ... : mouth crops (from the 512 crops) at times t0..t1
out, lmf, t0, t1, st = sys.argv[1], sys.argv[2], *map(float, sys.argv[3:6]); vids = sys.argv[6:]
LM = np.load(lmf)["lm"][..., :2]; ks = [int(round(t * 25)) for t in np.arange(t0, t1, st)]; rows = []
for v in vids:
    c = cv2.VideoCapture(v); F = {}; k = 0
    while k <= max(ks):
        ok, f = c.read()
        if not ok: break
        if k in ks: F[k] = f
        k += 1
    tiles = []
    for k in ks:
        p = LM[min(k, len(LM) - 1)]; cx, cy = p[[61, 291, 13, 14]].mean(0); r = 1.1 * np.linalg.norm(p[61] - p[291])
        x0, y0 = int(max(0, cx - r)), int(max(0, cy - 0.75 * r)); t = F.get(k, np.zeros((512, 512, 3), np.uint8))[y0:y0 + int(1.5 * r), x0:x0 + int(2 * r)]
        t = cv2.resize(t, (200, 150)); cv2.putText(t, f"{k/25:.2f}", (3, 14), 0, 0.45, (0, 255, 255), 1); tiles.append(t)
    rows.append(np.hstack(tiles))
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
