import sys, numpy as np, cv2
lm = np.load(sys.argv[1])["lm"][..., :2]; out = sys.argv[2]; frames = [int(x) for x in sys.argv[3].split(",")]; vids = sys.argv[4:]
rows = []
for k in frames:
    p = lm[k]; c = p[[61, 291, 13, 14]].mean(0); mw = np.linalg.norm(p[61] - p[291]); r = int(0.85 * mw)
    row = []
    for v in vids:
        lab, path = v.split("=", 1); cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(k + 1): ok, f = cap.read()
        crop = f[int(c[1] - 0.7 * r):int(c[1] + 0.7 * r), int(c[0] - r):int(c[0] + r)]; crop = cv2.resize(crop, (360, int(360 * crop.shape[0] / crop.shape[1])), interpolation=cv2.INTER_CUBIC)
        cv2.putText(crop, f"{lab} {k}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2); row.append(crop)
    h = min(x.shape[0] for x in row); rows.append(np.concatenate([x[:h] for x in row], 1))
w = min(x.shape[1] for x in rows); cv2.imwrite(out, np.concatenate([x[:, :w] for x in rows], 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
