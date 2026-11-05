#!/usr/bin/env python3
"""r15 F2: lip-line and teeth sharpness of a LatentSync layer (25 fps, master geometry) relative to the SOURCE itself.

Per frame, mean |Laplacian| of luma:
- lips  = the outer-lip hull minus the inner hull, dilated 0.04 mouth widths (the lip line and the lip texture);
- teeth = the inner-lip hull, only where opening / mouth width > --open (teeth / inner mouth visible).
Layer: only frames where it differs from the master (LatentSync frames), measured on the layer's own landmarks.
Source: every master frame, on the source landmarks. Reported as layer median / source median: 1.0 = as crisp as the source.
    python scripts/r15/layer_sharp.py --label X --layer L.mp4 --lm L.mediapipe.npz --src MASTER.mp4 --src-lm MASTER_MP.npz [--json out.json]
"""
import argparse, json
import numpy as np, cv2
LO = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LI = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


def measure(G, p, op_thr):
    if not np.isfinite(p[LO + LI]).all(): return None
    mw = float(np.linalg.norm(p[61] - p[291])); op = float(np.linalg.norm(p[13] - p[14])) / max(mw, 1e-6)
    x0, y0 = np.floor(p[LO].min(0) - 0.2 * mw).astype(int); x1, y1 = np.ceil(p[LO].max(0) + 0.2 * mw).astype(int)
    H, W = G.shape; x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 12 or y1 - y0 < 12: return None
    g = G[y0:y1, x0:x1].astype(np.float32); L = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    mo = np.zeros(g.shape, np.uint8); mi = np.zeros(g.shape, np.uint8)
    cv2.fillPoly(mo, [np.round(cv2.convexHull((p[LO] - [x0, y0]).astype(np.float32))[:, 0]).astype(np.int32)], 1)
    cv2.fillPoly(mi, [np.round(p[LI] - [x0, y0]).astype(np.int32)], 1)
    r = max(1, int(0.04 * mw)); k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    lips = (cv2.dilate(mo, k) > 0) & (cv2.erode(mi, k) == 0) if op > op_thr else cv2.dilate(mo, k) > 0
    teeth = (cv2.erode(mi, k) > 0) if op > op_thr else None
    return float(L[lips].mean()) if lips.any() else None, (float(L[teeth].mean()) if teeth is not None and teeth.sum() > 20 else None), (x0, y0, x1, y1, mo)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--label", required=True); ap.add_argument("--layer", required=True); ap.add_argument("--lm", required=True)
    ap.add_argument("--src", required=True); ap.add_argument("--src-lm", required=True); ap.add_argument("--open", type=float, default=0.15); ap.add_argument("--json")
    a = ap.parse_args(); lm = np.load(a.lm)["lm"][..., :2]; slm = np.load(a.src_lm)["lm"][..., :2]
    cl, cs = cv2.VideoCapture(a.layer), cv2.VideoCapture(a.src); Ls, Lt, Ss, St = [], [], [], []; k = 0
    while True:
        o1, fl = cl.read(); o2, fs = cs.read()
        if not (o1 and o2): break
        gl, gs = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fs, cv2.COLOR_BGR2GRAY)
        if k < len(slm):
            m = measure(gs, slm[k], a.open)
            if m: m[0] is not None and Ss.append(m[0]); m[1] is not None and St.append(m[1])
        if k < len(lm):
            m = measure(gl, lm[k], a.open)
            if m:
                x0, y0, x1, y1, mo = m[2]; d = np.abs(gl[y0:y1, x0:x1].astype(np.float32) - gs[y0:y1, x0:x1])[mo > 0]
                if d.size and d.mean() > 2.0:
                    m[0] is not None and Ls.append(m[0]); m[1] is not None and Lt.append(m[1])
        k += 1
    r = {"label": a.label, "ls_frames": len(Ls), "teeth_frames": len(Lt),
         "lips_vs_src": round(np.median(Ls) / np.median(Ss), 3) if Ls and Ss else None,
         "teeth_vs_src": round(np.median(Lt) / np.median(St), 3) if Lt and St else None,
         "lips_abs": round(float(np.median(Ls)), 2) if Ls else None, "teeth_abs": round(float(np.median(Lt)), 2) if Lt else None,
         "src_lips_abs": round(float(np.median(Ss)), 2) if Ss else None, "src_teeth_abs": round(float(np.median(St)), 2) if St else None}
    print("== layer_sharp", json.dumps(r), flush=True)
    if a.json: open(a.json, "a").write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
