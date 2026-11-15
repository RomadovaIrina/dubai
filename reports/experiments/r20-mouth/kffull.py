import sys, glob, os, cv2, numpy as np
# kffull.py out.jpg <dump root> <crop.mp4> f1,f2,.. tag1 tag2 ... : whole 512 keyframes (downscaled), rows = source + tags
out, root, crop, frames = sys.argv[1:5]; tags = sys.argv[5:]; frames = [int(x) for x in frames.split(",")]; c = cv2.VideoCapture(crop); rows = []
def lab(im, t):
    im = cv2.resize(im, (300, 300)); cv2.putText(im, t, (4, 14), 0, 0.45, (0, 0, 0), 3); cv2.putText(im, t, (4, 14), 0, 0.45, (0, 255, 255), 1); return im
r = []
for k in frames: c.set(cv2.CAP_PROP_POS_FRAMES, k); r.append(lab(c.read()[1], "src %d" % k))
rows.append(np.hstack(r))
for t in tags:
    F = {int(os.path.basename(f).split("_f")[1][:5]): f for f in glob.glob(os.path.join(root, t, "b*_s*_f*.jpg")) if "_s00_" not in f}
    rows.append(np.hstack([lab(cv2.imread(F[k]) if k in F else np.zeros((512, 512, 3), np.uint8), "%s %d" % (t, k)) for k in frames]))
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90]); print("wrote", out)
