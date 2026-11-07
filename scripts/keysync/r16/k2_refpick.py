#!/usr/bin/env python3
"""r16 K2: auto-pick of KeySync's identity / reference frame (KS_REF_IDX). A reference that shows the UPPER TEETH halves the black-hole mouths
(m03 13.8 -> 8.1 %, m18 23 -> 8 % of open frames) and raises SyncNet. Per frame of the KeySync input clip (the 512 crop, 25 fps), with MediaPipe
landmarks of that clip:
  open    inner-lip gap / mouth width, best 0.10-0.22 (slightly open; wide open = jaw shape baked into the identity)
  teeth   share of the UPPER half of the inner-lip hull that is teeth: brighter than 0.80 x the skin L and less red than the skin (a* < skin a*)
  dark    share of the inner-lip hull with L < 15 (black cavity) -> penalty
  sharp   mouth-box Laplacian variance / the clip median
  frontal |yaw| (nose tip vs the eye midpoint, / IOD) and |roll| (eye line angle)
  light   face-oval mean L / the clip median (too dark or blown out -> penalty)
score = 3 teeth + 1 open-fit - 2 dark + 0.5 min(sharp, 2) - 6 |yaw| - 2 |roll| - |light - 1|; top-3, at least 2 s apart.
Writes <out>.json (top-3 + per-frame features) and <out>.jpg (contact sheet: face + mouth zoom + scores) for a human override.
    python k2_refpick.py --video crop512.mp4 [--lm crop512.npz] --out refpick_m03"""
import argparse, json, subprocess, sys
import numpy as np, cv2

INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--lm", default=""); ap.add_argument("--out", required=True)
ap.add_argument("--gap", type=float, default=2.0); ap.add_argument("--open-target", type=float, default=0.16, dest="ot"); ap.add_argument("--topn", type=int, default=3); a = ap.parse_args()
if not a.lm:
    a.lm = a.out + ".mp.npz"
    subprocess.run(["/venv/mediapipe/bin/python", "/workspace/dub/dubai/scripts/quality/mp_landmarks.py", "--video", a.video, "--out", a.lm], check=True, stdout=subprocess.DEVNULL)
LM = np.load(a.lm)["lm"][..., :2].astype(np.float32)
c = cv2.VideoCapture(a.video); fps = c.get(cv2.CAP_PROP_FPS) or 25; R = []; k = 0
poly = lambda p, idx, sh: cv2.fillPoly(np.zeros(sh, np.uint8), [np.round(cv2.convexHull(p[idx].astype(np.float32))[:, 0]).astype(np.int32)], 1)
while True:
    ok, f = c.read()
    if not ok or k >= len(LM): break
    p = LM[k]; row = None
    if np.isfinite(p).all():
        sh = f.shape[:2]; lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32); L = lab[..., 0] * 100 / 255; A = lab[..., 1] - 128
        mw = float(np.linalg.norm(p[61] - p[291])); iod = float(np.linalg.norm(p[33] - p[263])); op = float(np.linalg.norm(p[13] - p[14])) / max(mw, 1)
        face = poly(p, OVAL, sh) & (1 - cv2.dilate(poly(p, LIPS, sh), np.ones((9, 9), np.uint8)))
        skL, skA = float(np.median(L[face > 0])), float(np.median(A[face > 0]))
        inner = cv2.erode(poly(p, INNER, sh), np.ones((3, 3), np.uint8))
        up = inner.copy(); up[int(round((p[13][1] + p[14][1]) / 2)):, :] = 0   # the upper half of the opening (under the upper lip)
        teeth = (L > 0.80 * skL) & (A < skA)
        t_up = float(teeth[up > 0].mean()) if up.sum() > 15 else 0.0; dark = float((L[inner > 0] < 15).mean()) if inner.sum() > 15 else 0.0
        x0, y0 = p[[61, 291, 0, 17]].min(0) - 0.15 * mw; x1, y1 = p[[61, 291, 0, 17]].max(0) + 0.15 * mw
        g = cv2.cvtColor(f[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)], cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(g, cv2.CV_32F).var()) if g.size else 0.0
        em = (p[33] + p[263]) / 2; ax = (p[263] - p[33]) / max(iod, 1)
        yaw = float(np.dot(p[1] - em, ax) / max(iod, 1)); roll = float(np.degrees(np.arctan2(ax[1], ax[0])))
        row = dict(k=k, t=round(k / fps, 2), open=round(op, 3), teeth_up=round(t_up, 3), dark=round(dark, 3), sharp=sharp, yaw=round(yaw, 3), roll=round(roll, 1), light=float(L[face > 0].mean()))
    if row: R.append(row)
    k += 1
sm = np.median([r["sharp"] for r in R]); lm_ = np.median([r["light"] for r in R])
for r in R:
    r["sharp_rel"] = round(r["sharp"] / max(sm, 1e-6), 2); r["light_rel"] = round(r["light"] / max(lm_, 1e-6), 2)
    ofit = 1.0 - min(abs(r["open"] - a.ot) / 0.06, 1.0) if r["open"] > 0.04 else 0.0
    r["score"] = round(3 * r["teeth_up"] + ofit - 2 * r["dark"] + 0.5 * min(r["sharp_rel"], 2) - 6 * abs(r["yaw"]) - 2 * abs(np.radians(r["roll"])) - abs(r["light_rel"] - 1), 3)
order = sorted(R, key=lambda r: -r["score"]); top = []
for r in order:
    if all(abs(r["t"] - q["t"]) >= a.gap for q in top): top.append(r)
    if len(top) == a.topn: break
cur = next((r for r in R if r["k"] == 0), None)
tiles = []
for lab_, r in [("frame0 (current)", cur)] + [(f"#{i + 1}", r) for i, r in enumerate(top)]:
    if r is None: continue
    c.set(cv2.CAP_PROP_POS_FRAMES, r["k"]); f = c.read()[1]; p = LM[r["k"]]; mw = np.linalg.norm(p[61] - p[291]); cx, cy = p[[61, 291, 13, 14]].mean(0)
    z = f[max(0, int(cy - 0.6 * mw)):int(cy + 0.6 * mw), max(0, int(cx - 0.9 * mw)):int(cx + 0.9 * mw)]
    t = np.vstack([cv2.resize(f, (300, 300)), cv2.resize(z, (300, 200))])
    for j, s in enumerate([f"{lab_}  k={r['k']} t={r['t']}s", f"score {r['score']}  teeth {r['teeth_up']}  open {r['open']}",
                           f"dark {r['dark']}  sharp {r['sharp_rel']}  light {r['light_rel']}", f"yaw {r['yaw']}  roll {r['roll']}"]):
        cv2.putText(t, s, (5, 18 + 18 * j), 0, 0.45, (0, 0, 0), 3); cv2.putText(t, s, (5, 18 + 18 * j), 0, 0.45, (0, 255, 255), 1)
    tiles.append(t)
cv2.imwrite(a.out + ".jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
json.dump({"video": a.video, "top": top, "frame0": cur}, open(a.out + ".json", "w"), indent=1)
print("K2_REFPICK " + json.dumps({"video": a.video, "pick": top[0]["k"] if top else 0, "top": [(r["k"], r["t"], r["score"], r["teeth_up"], r["open"]) for r in top],
                                  "frame0": (cur["score"], cur["teeth_up"], cur["open"]) if cur else None}))
