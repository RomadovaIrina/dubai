#!/usr/bin/env python3
"""r15 K: the "black hole" mouth. On open-mouth frames (inner-lip gap 13-14 / mouth width 61-291 > --open), the inner-lip hull's Lab: mean L,
the share of pixels with L < 12 (near black) and the share with L > 55 (teeth-bright); a frame is a "black hole" when mean L < 15 and bright < 2 %.
Per video: open frames, black-hole frames (count, share, times), the cavity mean Lab. Same 512 crop grid for all (source crop / KeySync outputs).
    python k_cavity.py --pairs label=video.mp4:landmarks.npz ... [--open 0.15]"""
import argparse, json
import numpy as np, cv2

INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
ap = argparse.ArgumentParser(); ap.add_argument("--pairs", nargs="+", required=True); ap.add_argument("--open", type=float, default=0.15); a = ap.parse_args()
out = {}
for pr in a.pairs:
    lb, rest = pr.split("=", 1); vid, lmf = rest.split(":", 1); LM = np.load(lmf)["lm"][..., :2].astype(np.float32)
    c = cv2.VideoCapture(vid); fps = c.get(cv2.CAP_PROP_FPS) or 25; k = 0; rows = []
    while True:
        ok, f = c.read()
        if not ok or k >= len(LM): break
        p = LM[k]
        if np.isfinite(p).all():
            gap = float(np.linalg.norm(p[13] - p[14])); mw = float(np.linalg.norm(p[61] - p[291]))
            if mw > 5 and gap / mw > a.open:
                mk = np.zeros(f.shape[:2], np.uint8); cv2.fillPoly(mk, [np.round(cv2.convexHull(p[INNER]))[:, 0].astype(np.int32)], 1)
                mk = cv2.erode(mk, np.ones((3, 3), np.uint8))
                if mk.sum() > 30:
                    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32); L = lab[..., 0][mk > 0] * 100 / 255
                    rows.append((k / fps, float(L.mean()), float((L < 12).mean()), float((L > 55).mean()), *[float(lab[..., i][mk > 0].mean()) - (128 if i else 0) for i in (1, 2)]))
        k += 1
    R = np.array(rows) if rows else np.zeros((0, 6)); bh = (R[:, 1] < 15) & (R[:, 3] < 0.02) if len(R) else np.zeros(0, bool)
    out[lb] = {"open_frames": int(len(R)), "black_hole_frames": int(bh.sum()), "black_hole_share": round(float(bh.mean()) if len(R) else 0, 3),
               "cavity_L_mean": round(float(R[:, 1].mean()), 1) if len(R) else None, "near_black_share": round(float(R[:, 2].mean()), 3) if len(R) else None,
               "teeth_bright_share": round(float(R[:, 3].mean()), 3) if len(R) else None, "cavity_ab": [round(float(R[:, 4].mean()), 1), round(float(R[:, 5].mean()), 1)] if len(R) else None,
               "black_hole_times": [round(float(t), 2) for t in R[bh, 0][:: max(1, int(bh.sum()) // 12 or 1)]] if bh.any() else []}
print(json.dumps(out, indent=1))
