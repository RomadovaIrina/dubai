# mdiff.py src.mp4 lm_src.npz fin.mp4 ...: per 0.5 s, mean |final - source| luma in the mouth box (is the drawn mouth there or the source?)
import sys, numpy as np, cv2
src, lmp = sys.argv[1:3]; L = np.load(lmp)["lm"][..., :2]
for fin in sys.argv[3:]:
    c1, c2 = cv2.VideoCapture(src), cv2.VideoCapture(fin); fps = c1.get(cv2.CAP_PROP_FPS); D = []
    for k in range(len(L)):
        o1, f1 = c1.read(); o2, f2 = c2.read()
        if not (o1 and o2): break
        q = L[k]; x0, x1 = int(q[61, 0] - 10), int(q[291, 0] + 10); y0, y1 = int(q[0, 1] - 15), int(q[17, 1] + 15)
        a = cv2.cvtColor(f1[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(float); b = cv2.cvtColor(f2[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(float); D.append(np.abs(a - b).mean())
    D = np.array(D); h = int(fps / 2); print(fin.split("/")[-1], " ".join(f"{np.mean(D[i:i + h]):.0f}" for i in range(0, len(D), h)))
