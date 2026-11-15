#!/usr/bin/env python3
"""keyframe sheet: rows = dump tags (+ the source crop), cols = keyframe frames. kfsheet.py out.jpg <dump root> <crop.mp4> <crop_lm.npz> f1,f2,... tag1 tag2 ...
prints per tag the black share (L < 6) inside the SOURCE-located mouth box over ALL its keyframes"""
import sys, glob, os, cv2, numpy as np
out, root, crop, lmf, frames = sys.argv[1:6]; tags = sys.argv[6:]; frames = [int(x) for x in frames.split(",")]
LM = np.load(lmf)["lm"][..., :2]; c = cv2.VideoCapture(crop)
def box(k):
    p = LM[min(k, len(LM) - 1)]
    if not np.isfinite(p).all(): return 156, 256, 200
    cx, cy = p[[61, 291, 0, 17]].mean(0); h = max(60, 0.8 * np.linalg.norm(p[61] - p[291])); return int(max(0, cx - h)), int(max(0, cy - 0.75 * h)), int(2 * h)
def tile(im, k, name):
    x, y, s = box(k); t = cv2.resize(im[y:y + int(0.8 * s), x:x + s], (300, 240), interpolation=cv2.INTER_CUBIC)
    cv2.putText(t, "%s %d" % (name, k), (4, 14), 0, 0.45, (0, 0, 0), 3); cv2.putText(t, "%s %d" % (name, k), (4, 14), 0, 0.45, (0, 255, 255), 1); return t
rows = []; r = []
for k in frames:
    c.set(cv2.CAP_PROP_POS_FRAMES, k); r.append(tile(c.read()[1], k, "src"))
rows.append(np.hstack(r))
for t in tags:
    F = {int(os.path.basename(f).split("_f")[1][:5]): f for f in glob.glob(os.path.join(root, t, "b*_s*_f*.jpg")) if "_s00_" not in f}
    rows.append(np.hstack([tile(cv2.imread(F[k]) if k in F else np.zeros((512, 512, 3), np.uint8), k, t) for k in frames]))
    bl = []; dk = []
    for k, f in sorted(F.items()):
        if k >= len(LM) or not np.isfinite(LM[k]).all(): continue
        p = LM[k]; x0, y0 = p[[61, 291, 0, 17]].min(0); x1, y1 = p[[61, 291, 0, 17]].max(0); im = cv2.imread(f)
        L = cv2.cvtColor(im[int(y0):int(y1), int(x0):int(x1)].astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[..., 0]
        if L.size: bl.append(float((L < 6).mean())); dk.append(float((L < 15).mean()))
    print("KF %-10s keyframes %3d | black (L<6) share in the mouth box: mean %.3f, keyframes > 0.10: %2d | dark (L<15): mean %.3f" % (t, len(bl), np.mean(bl), int((np.array(bl) > 0.10).sum()), np.mean(dk)))
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90]); print("wrote", out)
