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
score = 3 teeth + 1 open-fit(0.18) - 2 smile (corner lift > speaker median + 0.03) - wide (width > 1.15 x median) - 2 occl (lower-face patch outlier: hand / cup) - 4 max(0, 0.85 - face size / median) - 8 max(0, centre distance from the clip median / face height - 0.08) - 2 dark + 0.5 min(sharp, 2) - 6 |yaw| - 2 |roll| - |light - 1|; top-N at least --gap s apart, each checked with MediaPipe Hands (a hand inside the face box -> rejected).
Presence (--face-lm, + --box): only frames where THIS face is present in the master track (finite landmarks; not a bridged / absent frame) AND the
crop landmarks are that face (their centre within 0.25 face heights of the master track projected through the crop box) are candidates, so the
reference is never another person / an empty crop (audit orch. 10).
Writes <out>.json (top-3 + per-frame features) and <out>.jpg (contact sheet: face + mouth zoom + scores) for a human override.
    python k2_refpick.py --video crop512.mp4 [--lm crop512.npz] --out refpick_m03"""
import argparse, json, subprocess, sys
import numpy as np, cv2

INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--lm", default=""); ap.add_argument("--out", required=True)
ap.add_argument("--gap", type=float, default=2.0); ap.add_argument("--open-target", type=float, default=0.18, dest="ot"); ap.add_argument("--topn", type=int, default=3)
ap.add_argument("--face-lm", default="", dest="face_lm", help="the face MASTER landmarks (lm_<tag>.npz: NaN where absent), frame-aligned with --video")
ap.add_argument("--box", default="", help="the crop box json (static x/y/side or tracked xs/ys) mapping the master to --video"); a = ap.parse_args()
import os
HERE = os.path.dirname(os.path.realpath(__file__))
if not a.lm:
    a.lm = a.out + ".mp.npz"
    mpl = next((q for q in [os.path.join(HERE, "../../quality/mp_landmarks.py"), os.path.join(HERE, "mp_landmarks.py"),
                             os.path.join(os.environ.get("DABAI_ROOT", "."), "scripts/quality/mp_landmarks.py")] if os.path.exists(q)), None)
    if mpl is None: sys.exit("k2_refpick: no --lm and mp_landmarks.py not found (set DABAI_ROOT)")
    subprocess.run([os.environ.get("DUB_MEDIAPIPE_PY", sys.executable), mpl, "--video", a.video, "--out", a.lm], check=True, stdout=subprocess.DEVNULL)
LM = np.load(a.lm)["lm"][..., :2].astype(np.float32)
PRES = None   # per crop frame: this face is present and the crop landmarks are it
if a.face_lm:
    FL = np.load(a.face_lm)["lm"][..., :2].astype(np.float32); PRES = np.zeros(len(LM), bool)
    B = json.load(open(a.box)) if a.box else None
    for k in range(min(len(LM), len(FL))):
        q = FL[k]
        if not (np.isfinite(q).all() and np.isfinite(LM[k]).all()): continue
        if B is None: PRES[k] = True; continue
        x0 = B["xs"][min(k, len(B["xs"]) - 1)] if "xs" in B else B["x"]; y0 = B["ys"][min(k, len(B["ys"]) - 1)] if "ys" in B else B["y"]; sc = 512.0 / B["side"]
        cq = (q.mean(0) - (x0, y0)) * sc; fhq = float(np.linalg.norm(q[10] - q[152])) * sc
        PRES[k] = np.hypot(*(LM[k].mean(0) - cq)) < 0.25 * max(fhq, 1)
    print(f"k2_refpick: presence {int(PRES.sum())} / {len(PRES)} frames", file=sys.stderr)
    if PRES.sum() == 0: sys.exit("k2_refpick: the face is never present in the crop")
c = cv2.VideoCapture(a.video); fps = c.get(cv2.CAP_PROP_FPS) or 25; R = []; k = 0; PATCH = []
poly = lambda p, idx, sh: cv2.fillPoly(np.zeros(sh, np.uint8), [np.round(cv2.convexHull(p[idx].astype(np.float32))[:, 0]).astype(np.int32)], 1)
while True:
    ok, f = c.read()
    if not ok or k >= len(LM): break
    p = LM[k]; row = None
    if np.isfinite(p).all() and (PRES is None or PRES[k]):
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
        nrm = np.array([-ax[1], ax[0]]) if False else None
        ax2 = (p[291] - p[61]) / max(mw, 1); nr = np.array([-ax2[1], ax2[0]])
        lift = float(np.dot((p[0] + p[17]) / 2 - (p[61] + p[291]) / 2, nr) / max(mw, 1))   # > 0: the corners above the lip centre (smile)
        # occlusion probe: the lower face (nose base -> chin, cheek to cheek) minus the lips, warped to a canonical 48 x 32 patch (a hand / cup changes it)
        src3 = np.float32([p[234], p[454], p[152]]); dst3 = np.float32([[0, 0], [47, 0], [24, 31]])
        lf = cv2.warpAffine(cv2.cvtColor(f, cv2.COLOR_BGR2LAB), cv2.getAffineTransform(src3, dst3), (48, 32)).astype(np.float32)
        lipm = cv2.warpAffine(poly(p, LIPS, sh).astype(np.float32), cv2.getAffineTransform(src3, dst3), (48, 32)) < 0.5
        PATCH.append((k, lf, lipm))
        lh = poly(p, LIPS, sh) & (1 - poly(p, INNER, sh)); lipc = float(np.median(A[lh > 0]) - skA) if lh.sum() > 20 else 0.0   # the lips are redder than the skin; a hand over them is not
        row = dict(k=k, t=round(k / fps, 2), open=round(op, 3), lift=lift, width=mw / max(iod, 1), fh=float(np.linalg.norm(p[10] - p[152])), lipc=lipc, cx=float(p[:, 0].mean()), cy=float(p[:, 1].mean()), teeth_up=round(t_up, 3), dark=round(dark, 3), sharp=sharp, yaw=round(yaw, 3), roll=round(roll, 1), light=float(L[face > 0].mean()))
    if row: R.append(row)
    k += 1
MEDP = np.median(np.stack([q[1] for q in PATCH[:: max(1, len(PATCH) // 300)]]), 0); occ = {}
for kk, lf, lipm in PATCH: occ[kk] = float(np.abs(lf - MEDP)[lipm].mean())
mlc = max(np.median([r["lipc"] for r in R]), 1e-3); mcx, mcy = np.median([r["cx"] for r in R]), np.median([r["cy"] for r in R])
ov = np.array(list(occ.values())); o75, o95 = np.percentile(ov, 75), np.percentile(ov, 95); mfh = np.median([r["fh"] for r in R])
for r in R:
    r["pos"] = round(float(np.hypot(r["cx"] - mcx, r["cy"] - mcy) / max(r["fh"], 1)), 3)   # KeySync assumes the identity face roughly where the frames have it (q02: a far-off reference -> garbage mouth)
    r["occl"] = round(float(max(np.clip((occ[r["k"]] - o75) / max(o95 - o75, 1e-6), 0, 2), np.clip((0.7 - r["lipc"] / mlc) / 0.3, 0, 2))), 2); r["fh_rel"] = round(r["fh"] / mfh, 2)
sm = np.median([r["sharp"] for r in R]); lm_ = np.median([r["light"] for r in R]); ml = np.median([r["lift"] for r in R]); mwd = np.median([r["width"] for r in R])
for r in R:
    r["sharp_rel"] = round(r["sharp"] / max(sm, 1e-6), 2); r["light_rel"] = round(r["light"] / max(lm_, 1e-6), 2)
    ofit = 1.0 - min(abs(r["open"] - a.ot) / 0.08, 1.0) if r["open"] > 0.04 else 0.0
    r["smile"] = round(max(0.0, (r["lift"] - ml) - 0.03) / 0.03, 2); r["wide"] = round(max(0.0, r["width"] / mwd - 1.15) / 0.10, 2)   # vs the speaker's own median
    r["lift_rel"] = round(r["lift"] - ml, 3); r["width_rel"] = round(r["width"] / mwd, 3)
    r["score"] = round(3 * r["teeth_up"] + ofit - 2 * min(r["smile"], 2) - min(r["wide"], 2) - 2 * r["occl"] - 4 * max(0.0, 0.85 - r["fh_rel"]) - 8 * max(0.0, r["pos"] - 0.08) - 2 * r["dark"] + 0.5 * min(r["sharp_rel"], 2) - 6 * abs(r["yaw"]) - 2 * abs(np.radians(r["roll"])) - abs(r["light_rel"] - 1), 3)
order = sorted(R, key=lambda r: -r["score"]); top = []; HANDS = None; rej = []
def hand_over_face(k):   # MediaPipe Hands on a shortlisted frame: any hand landmark inside the face box (+10 %) -> reject (a hand / finger over the face)
    global HANDS
    import mediapipe as mp
    if HANDS is None: HANDS = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.4)
    c.set(cv2.CAP_PROP_POS_FRAMES, k); f = c.read()[1]; h, w = f.shape[:2]; res = HANDS.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks: return False
    p = LM[k]; x0, y0 = p.min(0); x1, y1 = p.max(0); m = 0.1 * (x1 - x0)
    return any(x0 - m <= l.x * w <= x1 + m and y0 - m <= l.y * h <= y1 + m for hl in res.multi_hand_landmarks for l in hl.landmark)
for r in order:
    if all(abs(r["t"] - q["t"]) >= a.gap for q in top):
        if hand_over_face(r["k"]): rej.append(r["k"]); continue
        top.append(r)
    if len(top) == a.topn: break
cur = next((r for r in R if r["k"] == 0), None)
tiles = []
for lab_, r in [("frame0 (current)", cur)] + [(f"#{i + 1}", r) for i, r in enumerate(top)]:
    if r is None: continue
    c.set(cv2.CAP_PROP_POS_FRAMES, r["k"]); f = c.read()[1]; p = LM[r["k"]]; mw = np.linalg.norm(p[61] - p[291]); cx, cy = p[[61, 291, 13, 14]].mean(0)
    z = f[max(0, int(cy - 0.6 * mw)):int(cy + 0.6 * mw), max(0, int(cx - 0.9 * mw)):int(cx + 0.9 * mw)]
    t = np.vstack([cv2.resize(f, (300, 300)), cv2.resize(z, (300, 200))])
    for j, s in enumerate([f"{lab_}  k={r['k']} t={r['t']}s", f"score {r['score']}  teeth {r['teeth_up']}  open {r['open']}",
                           f"dark {r['dark']}  sharp {r['sharp_rel']}  smile {r['smile']} lift {r['lift_rel']}", f"yaw {r['yaw']} roll {r['roll']} occl {r['occl']} size {r['fh_rel']}"]):
        cv2.putText(t, s, (5, 18 + 18 * j), 0, 0.45, (0, 0, 0), 3); cv2.putText(t, s, (5, 18 + 18 * j), 0, 0.45, (0, 255, 255), 1)
    tiles.append(t)
cv2.imwrite(a.out + ".jpg", np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
json.dump({"video": a.video, "top": top, "frame0": cur, "rejected_hand": rej}, open(a.out + ".json", "w"), indent=1)
print("K2_REFPICK " + json.dumps({"video": a.video, "pick": top[0]["k"] if top else 0, "top": [(r["k"], r["t"], r["score"], r["teeth_up"], r["open"], r["lift_rel"], r["width_rel"]) for r in top],
                                  "frame0": (cur["score"], cur["teeth_up"], cur["open"]) if cur else None, "rejected_hand": rej}))
