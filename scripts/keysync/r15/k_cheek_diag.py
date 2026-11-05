#!/usr/bin/env python3
"""r15 K: where does a KeySync final change pixels it should not? Per source frame, mean |final - source| (grey) in:
  other   = the other faces' hulls (the child), dilated 0.1 of the other face's IOD;
  band    = between the speaker's face oval and the other face (outside the speaker's oval dilated 0.05 IOD, inside the paste box, x on the other face's side);
  chin    = under the speaker's jaw: the oval's lower part dilated 0.25 IOD minus the oval;
the top frames (with times) of 'other' and 'band' = the "cheek slides into the child" moments; 'chin' for the chin note.
    python k_cheek_diag.py --final F.mp4 --src SRC --faces faces2.npz --speaker-lm src.mediapipe.npz --box box.json [--label x]"""
import argparse, json
import numpy as np, cv2

OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
JAW = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234]
ap = argparse.ArgumentParser()
for k in ("final", "src", "faces", "speaker_lm", "box"): ap.add_argument("--" + k.replace("_", "-"), required=True)
ap.add_argument("--label", default="")
a = ap.parse_args()
F2 = np.load(a.faces)["others"]; SP = np.load(a.speaker_lm)["lm"][..., :2].astype(np.float32); b = json.load(open(a.box))
cf, cs = cv2.VideoCapture(a.final), cv2.VideoCapture(a.src); fps = cs.get(cv2.CAP_PROP_FPS) or 30.0
R = {"other": [], "band": [], "chin": []}; T = []; k = 0
hull = lambda pts: np.round(cv2.convexHull(pts.astype(np.float32))[:, 0]).astype(np.int32)
while True:
    okf, ff = cf.read(); oks, fs = cs.read()
    if not (okf and oks): break
    p = SP[min(k, len(SP) - 1)]; o = F2[min(k, len(F2) - 1)]
    if np.isfinite(p).all():
        H, W = fs.shape[:2]; g = np.abs(cv2.cvtColor(ff, cv2.COLOR_BGR2GRAY).astype(np.float32) - cv2.cvtColor(fs, cv2.COLOR_BGR2GRAY).astype(np.float32))
        iod = float(np.linalg.norm(p[33] - p[263])); bx = np.zeros((H, W), np.uint8); bx[b["y"]:b["y"] + b["side"], b["x"]:b["x"] + b["side"]] = 1
        ov = np.zeros((H, W), np.uint8); cv2.fillPoly(ov, [hull(p[OVAL])], 1)
        ovd = cv2.dilate(ov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.05 * iod) + 1,) * 2))
        jw = np.zeros((H, W), np.uint8); cv2.fillPoly(jw, [hull(p[JAW])], 1); jw = cv2.dilate(jw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.25 * iod) + 1,) * 2))
        chin = (jw > 0) & (ov == 0) & (np.arange(H)[:, None] > p[152, 1] - 0.2 * iod)
        R["chin"].append(float(g[chin & (bx > 0)].mean()) if (chin & (bx > 0)).any() else 0.0)
        om = np.zeros((H, W), np.uint8)
        for f2 in o:
            if np.isfinite(f2).all():
                oi = float(np.linalg.norm(f2[33] - f2[263])); m1 = np.zeros((H, W), np.uint8); cv2.fillPoly(m1, [hull(f2[OVAL])], 1)
                om |= cv2.dilate(m1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.1 * oi) + 1,) * 2))
        sel = (om > 0) & (bx > 0) & (ov == 0); R["other"].append(float(g[sel].mean()) if sel.sum() > 50 else 0.0)
        side = -1 if any(np.isfinite(f2).all() and f2.mean(0)[0] < p.mean(0)[0] for f2 in o) else 1   # the other face's side
        xs = np.arange(W)[None, :]; sm = (xs < p[:, 0].min() + 0.2 * iod) if side < 0 else (xs > p[:, 0].max() - 0.2 * iod)
        band = (bx > 0) & (ovd == 0) & sm & (np.arange(H)[:, None] > p[:, 1].min()) & (np.arange(H)[:, None] < p[152, 1] + 0.3 * iod)
        R["band"].append(float(g[band].mean()) if band.sum() > 50 else 0.0); T.append(k / fps)
    k += 1
out = {"label": a.label, "frames": len(T)}
for r, v in R.items():
    v = np.array(v); top = np.argsort(-v)[:8]
    out[r] = {"mean": round(float(v.mean()), 2), "p95": round(float(np.percentile(v, 95)), 2), "top": [[round(T[i], 2), round(float(v[i]), 1)] for i in sorted(top)]}
print(json.dumps(out))
