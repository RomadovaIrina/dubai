import sys, numpy as np, cv2
INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
CHEEK = [50, 280]   # cheek centres (landmarks 50 / 280), 9x9 px patches
def run(lab, vid, lmf, n=None):
    L = np.load(lmf)["lm"][..., :2]; c = cv2.VideoCapture(vid); k = 0; R = []; C = []
    while True:
        ok, f = c.read()
        if not ok or k >= len(L) or (n and k >= n): break
        p = L[k]
        if np.isfinite(p).all():
            w = np.linalg.norm(p[61] - p[291]); g = np.linalg.norm(p[13] - p[14]) / w
            if g > 0.12:
                m = cv2.fillPoly(np.zeros(f.shape[:2], np.uint8), [cv2.convexHull(p[INNER].astype(np.float32))[:, 0].astype(np.int32)], 1)
                m = cv2.erode(m, np.ones((3, 3), np.uint8))
                Lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32); Lab[..., 0] *= 100 / 255; Lab[..., 1:] -= 128
                if m.sum() > 20:
                    px = Lab[m > 0]; dark = px[px[:, 0] < 30]
                    R.append((px[:, 0].mean(), px[:, 1].mean(), px[:, 2].mean(), dark[:, 2].mean() if len(dark) > 5 else np.nan, (px[:, 2] < -3).mean()))
                for ci in CHEEK:
                    x, y = p[ci].astype(int); C.append(Lab[max(0, y - 4):y + 5, max(0, x - 4):x + 5].reshape(-1, 3).mean(0))
        k += 1
    R = np.array(R); C = np.array(C)
    print(f"{lab:7s} open frames {len(R):4d} | mouth L {np.nanmean(R[:,0]):5.1f} a {np.nanmean(R[:,1]):5.1f} b {np.nanmean(R[:,2]):5.1f} | dark (L<30) b {np.nanmean(R[:,3]):5.1f} | blue px (b<-3) {np.nanmean(R[:,4])*100:4.1f} % | cheek L {C[:,0].mean():.1f} a {C[:,1].mean():.1f} b {C[:,2].mean():.1f}")
run("source", "w/ks/master_25fps.mp4", "src.npz")
run("raw", "w/ks/out/f0/pad_f0_f0.mp4", "raw.npz", 841)
run("guard", "w/ks/post_f0/guard.mp4", "guard.npz", 841)
run("final", "s3.mp4", "final.npz")
