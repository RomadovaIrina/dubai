import sys, numpy as np, cv2
# sharp.py lm.npz vidA vidB : per 0.2 s, mouth-box Laplacian var (A, B, A/B)
L = np.load(sys.argv[1])["lm"][..., :2]; ca, cb = cv2.VideoCapture(sys.argv[2]), cv2.VideoCapture(sys.argv[3]); R = []
for k in range(150):
    oa, fa = ca.read(); ob, fb = cb.read()
    if not (oa and ob): break
    p = L[k]; cx, cy = p[[61, 291, 13, 14]].mean(0); r = 0.7 * np.linalg.norm(p[61] - p[291])
    sl = (slice(int(cy - 0.6 * r), int(cy + 0.6 * r)), slice(int(cx - r), int(cx + r)))
    v = [cv2.Laplacian(cv2.cvtColor(f[sl], cv2.COLOR_BGR2GRAY), cv2.CV_32F).var() for f in (fa, fb)]; R.append(v)
R = np.array(R)
for i in range(0, len(R), 5): print(f"{i/25:.1f} {R[i:i+5,0].mean():7.1f} {R[i:i+5,1].mean():7.1f} {R[i:i+5,0].mean()/R[i:i+5,1].mean():.2f}")
