import sys, cv2, numpy as np
# zsheet.py out.jpg x,y,w,h scale t1,t2,... name=video ...   (frames by time, a fixed crop, upscaled)
out = sys.argv[1]; x, y, w, h = [int(v) for v in sys.argv[2].split(",")]; sc = float(sys.argv[3]); ts = [float(v) for v in sys.argv[4].split(",")]; rows = []
for a in sys.argv[5:]:
    name, p = a.split("=", 1); c = cv2.VideoCapture(p); fps = c.get(cv2.CAP_PROP_FPS) or 25; r = []
    for t in ts:
        c.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps))); ok, f = c.read()
        if not ok: f = np.zeros((h + y, w + x, 3), np.uint8)
        z = cv2.resize(f[y:y + h, x:x + w], None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        cv2.putText(z, "%s %.2f" % (name, t), (4, 14), 0, 0.45, (0, 0, 0), 3); cv2.putText(z, "%s %.2f" % (name, t), (4, 14), 0, 0.45, (0, 255, 255), 1); r.append(z)
    rows.append(np.hstack(r))
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92]); print("wrote", out)
