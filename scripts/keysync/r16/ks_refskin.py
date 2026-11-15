#!/usr/bin/env python3
"""r19 (04, a selfie under trees: grey stubble above the lip + a dark, green first frame -> KeySync drew a smooth dark-green band above the lip for the
whole video; with a reference frame whose upper-lip skin is lit like the cheeks the band is gone): pick KeySync's identity / reference frame by the
SKIN ABOVE THE LIP. Per frame of the KeySync input clip (the 512 crop) with its MediaPipe landmarks:
  closed    inner-lip gap / mouth width < --open (0.04): a neutral mouth, as frame 0 usually is (not K2's teeth frame)
  frontal   |yaw| < --yaw (0.15)
  dL        cheek L - upper-lip-skin L (the shadow / stubble shade above the lip), da = upper-lip a* - cheek a* (negative = greener than the cheeks)
score = -dL + 0.5 da + 0.05 cheek L. The pick replaces frame 0 only if it beats frame 0's score by --gain (2.0); otherwise 0 (the default stays).
    ks_refskin.py --video crop512.mp4 --lm crop512_lm.npz [--face-lm lm_f0.npz]   ->   prints KS_REFSKIN {"pick": k, ...}"""
import argparse, json, subprocess
import numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--face-lm", default="", dest="face_lm")
ap.add_argument("--open", type=float, default=0.04); ap.add_argument("--yaw", type=float, default=0.15); ap.add_argument("--gain", type=float, default=2.0); a = ap.parse_args()
pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", a.video], capture_output=True, text=True).stdout.strip().split(","); W, H = int(pr[0]), int(pr[1]); FB = W * H * 3
LM = np.load(a.lm)["lm"][..., :2].astype(np.float32); N = len(LM); ok = np.isfinite(LM.reshape(N, -1)).all(1)
if a.face_lm:   # only frames where THIS face is present in the master track
    fl = np.load(a.face_lm); fm = fl["lm"][..., :2]; J = fl["j"] if "j" in fl.files else np.arange(len(fm))
    ok &= np.array([np.isfinite(fm[min(int(J[min(k, len(J) - 1)]), len(fm) - 1)]).all() for k in range(N)])
CH = [[116, 123, 147, 187, 205, 36, 100, 117], [345, 352, 376, 411, 425, 266, 329, 346]]; PH = [98, 327, 2, 164, 0, 37, 267, 39, 269, 165, 391]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
hull = lambda p: cv2.convexHull(p.astype(np.float32))[:, 0].astype(np.int32)
p_ = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", a.video, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=FB * 2); R = {}
for k in range(N):
    b = p_.stdout.read(FB)
    if len(b) < FB: break
    if not ok[k]: continue
    p = LM[k]; lab = cv2.cvtColor(np.frombuffer(b, np.uint8).reshape(H, W, 3).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    cm = np.zeros((H, W), np.uint8); [cv2.fillPoly(cm, [hull(p[r])], 1) for r in CH]
    pm = np.zeros((H, W), np.uint8); cv2.fillPoly(pm, [hull(p[PH])], 1); lm_ = np.zeros((H, W), np.uint8); cv2.fillPoly(lm_, [hull(p[LIPS])], 1); pm = pm * (1 - cv2.dilate(lm_, np.ones((5, 5), np.uint8)))
    if pm.sum() < 30 or cm.sum() < 30: continue
    c, q = np.median(lab[cm > 0], 0), np.median(lab[pm > 0], 0); op = float(np.linalg.norm(p[13] - p[14]) / max(float(np.linalg.norm(p[61] - p[291])), 1e-3))
    dl = float(np.linalg.norm(p[1] - p[234])); dr = float(np.linalg.norm(p[454] - p[1])); yaw = abs(dl - dr) / max(dl + dr, 1.0)
    R[k] = dict(cheek_L=float(c[0]), dL=float(c[0] - q[0]), da=float(q[1] - c[1]), open=op, yaw=yaw, score=float(-(c[0] - q[0]) + 0.5 * (q[1] - c[1]) + 0.05 * c[0]))
C = [k for k, r in R.items() if r["open"] < a.open and r["yaw"] < a.yaw]; pick = 0; why = "no closed frontal frame: 0"
if C:
    best = max(C, key=lambda k: R[k]["score"]); s0 = R[0]["score"] if 0 in R else -1e9
    if R[best]["score"] - s0 >= a.gain: pick = best; why = "upper-lip skin closer to the cheeks than in frame 0"
    else: why = "frame 0 is as good (within %.1f): 0" % a.gain
rnd = lambda r: {k: round(v, 2) for k, v in r.items()}
print("KS_REFSKIN " + json.dumps({"pick": int(pick), "why": why, "frame0": rnd(R[0]) if 0 in R else None, "picked": rnd(R[pick]) if pick in R else None, "candidates": len(C), "frames": N}), flush=True)
