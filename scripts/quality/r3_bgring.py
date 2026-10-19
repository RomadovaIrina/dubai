#!/usr/bin/env python3
"""Round 3 D: background touched next to the face (user: "the background beside the cheek wobbles"). Per frame, a ring 5-25 px OUTSIDE
the MediaPipe face contour (FACE_OVAL of the reference frame's landmarks): mean |out - ref| (grey levels) and the share of ring pixels
changed by > 4 levels. ref = the same-instant reference with the same frame count (the source for source-fps outputs, the 25 fps master
for 25 fps outputs). Untouched background = encoder noise only (~0.3-0.8).
    python scripts/quality/r3_bgring.py --video OUT.mp4 --ref REF.mp4 --lm REF_landmarks.npz [--out x.json] [--inner 5 --outer 25]"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]


def main() -> int:
    import av, cv2
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--lm", required=True)
    ap.add_argument("--out", default=None); ap.add_argument("--inner", type=int, default=5); ap.add_argument("--outer", type=int, default=25)
    a = ap.parse_args(); lm = np.load(a.lm)["lm"]
    cv_, cr = av.open(a.video), av.open(a.ref); dv, dr = cv_.decode(cv_.streams.video[0]), cr.decode(cr.streams.video[0])
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.inner + 1, 2 * a.inner + 1)); k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.outer + 1, 2 * a.outer + 1))
    per = []; frac = []; lo = []; up = []; k = 0
    for fv, fr in zip(dv, dr):
        if k < len(lm) and np.isfinite(lm[k][FACE_OVAL]).all():
            o = fv.to_ndarray(format="gray").astype(np.float32); r = fr.to_ndarray(format="gray").astype(np.float32)
            m = np.zeros(o.shape, np.uint8); cv2.fillPoly(m, [lm[k][FACE_OVAL, :2].astype(np.int32)], 1)
            ring = (cv2.dilate(m, k2) > 0) & ~(cv2.dilate(m, k1) > 0)
            if ring.sum() > 50:
                d = np.abs(o - r)[ring]; per.append(float(d.mean())); frac.append(float((d > 4).mean()))
                yy = np.arange(o.shape[0])[:, None] > float(lm[k][1, 1])   # below the nose tip = cheeks / jaw (pasted region side); above = never pasted
                dl, du = np.abs(o - r)[ring & yy], np.abs(o - r)[ring & ~yy]
                if dl.size > 30 and du.size > 30: lo.append(float(dl.mean())); up.append(float(du.mean()))
        k += 1
    cv_.close(); cr.close()
    per = np.array(per); frac = np.array(frac)
    r = {"video": a.video, "ref": a.ref, "frames": int(len(per)), "ring_px": [a.inner, a.outer], "ring_absdiff_mean": round(float(per.mean()), 3) if len(per) else None,
         "ring_absdiff_p95": round(float(np.percentile(per, 95)), 3) if len(per) else None, "ring_changed_gt4_share": round(float(frac.mean()), 4) if len(frac) else None,
         "lower_ring_mean": round(float(np.mean(lo)), 3) if lo else None, "upper_ring_mean (noise floor)": round(float(np.mean(up)), 3) if up else None,
         "lower_minus_upper": round(float(np.mean(lo) - np.mean(up)), 3) if lo else None, "lower_minus_upper_p95": round(float(np.percentile(np.array(lo) - np.array(up), 95)), 3) if lo else None}
    print(json.dumps(r))
    if a.out: pathlib.Path(a.out).write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
