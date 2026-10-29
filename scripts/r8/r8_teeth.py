#!/usr/bin/env python3
"""r8 F: teeth / cavity visibility per video (own MediaPipe landmarks, cached as <stem>.flk_mp.npz by r8_mouth_flicker.py): over open-mouth frames
(inner opening > 0.08 IOD): teeth fraction = share of inner-lip-polygon pixels with luma > 140, cavity luma = median inner luma, dark-hole
share = frames whose inner 10th-percentile luma < 20. Also prints the median face centre / size (for a zoom crop).
    python scripts/r8/r8_teeth.py label:video.mp4 ...
"""
import sys, json, pathlib, numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from r8_mouth_flicker import LIPS_INNER, landmarks

res = {}
for spec in sys.argv[1:]:
    lab, vid = spec.split(":", 1); vid = pathlib.Path(vid); lm = landmarks(vid, None); c = cv2.VideoCapture(str(vid)); k = 0
    teeth, cav, dark, cen = [], [], [], []
    while True:
        ok, f = c.read()
        if not ok or k >= len(lm): break
        p = lm[k]; k += 1
        if not np.isfinite(p[LIPS_INNER + [33, 263, 13, 14, 10, 152]]).all(): continue
        iod = np.linalg.norm(p[33, :2] - p[263, :2]); cen.append([*p[[10, 152], :2].mean(0), np.linalg.norm(p[10, :2] - p[152, :2])])
        if np.linalg.norm(p[13, :2] - p[14, :2]) / iod < 0.08: continue
        m = np.zeros(f.shape[:2], np.uint8); cv2.fillPoly(m, [np.round(p[LIPS_INNER, :2]).astype(np.int32)], 1)
        Y = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[m > 0]
        if Y.size < 12: continue
        teeth.append((Y > 140).mean()); cav.append(np.median(Y)); dark.append(np.percentile(Y, 10) < 20)
    cen = np.array(cen)
    res[lab] = {"open_frames": len(teeth), "teeth_frac": round(float(np.mean(teeth)), 3) if teeth else None, "cavity_luma_med": round(float(np.median(cav)), 1) if cav else None,
                "dark_hole_frames_pct": round(100 * float(np.mean(dark)), 1) if dark else None,
                "face_cx": round(float(np.median(cen[:, 0])), 1) if len(cen) else None, "face_cy": round(float(np.median(cen[:, 1])), 1) if len(cen) else None,
                "face_h": round(float(np.median(cen[:, 2])), 1) if len(cen) else None}
    print("==", lab, json.dumps(res[lab]), flush=True)
