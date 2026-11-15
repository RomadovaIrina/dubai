#!/usr/bin/env python3
"""r20: where is the pasted lower face a smear? Per master frame, inside the lower-face polygon (nose base -> chin, cheek to cheek) of the SOURCE landmarks:
  hf   = detail energy (Laplacian std, luma) of the final / of the source   (a smear: << 1)
  mth  = the same inside the mouth box only
  diff = mean |final - source| luma (0 = the source face was kept)
  fw   = face width in px (landmarks 234 - 454)
smear.py --src master_25fps.mp4 --final final.mp4 --lm lm_f0.npz [--name x] -> one summary line + ranges (>= 0.4 s) where diff > 2 and hf < --thr"""
import argparse, subprocess, json
import numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--final", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--name", default="x")
ap.add_argument("--thr", type=float, default=0.7); ap.add_argument("--out", default=""); a = ap.parse_args()
z = np.load(a.lm); LM = z["lm"][..., :2].astype(np.float32); J = z["j"] if "j" in z.files else np.arange(len(LM))
pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", a.src], capture_output=True, text=True).stdout.strip().split(","); W, H = int(pr[0]), int(pr[1]); FB = W * H
rd = lambda p: subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", p, "-vf", f"fps=25,scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE, bufsize=FB * 4)
ps, pf = rd(a.src), rd(a.final); R = []
LOW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 2]
for k in range(len(J)):
    bs, bf = ps.stdout.read(FB), pf.stdout.read(FB)
    if len(bs) < FB or len(bf) < FB: break
    p = LM[min(int(J[k]), len(LM) - 1)]
    if not np.isfinite(p).all(): R.append((k, np.nan, np.nan, np.nan, np.nan)); continue
    s = np.frombuffer(bs, np.uint8).reshape(H, W).astype(np.float32); f = np.frombuffer(bf, np.uint8).reshape(H, W).astype(np.float32)
    m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [cv2.convexHull(p[LOW])[:, 0].astype(np.int32)], 1)
    mw = float(np.linalg.norm(p[61] - p[291])); x0, y0 = (p[[61, 291, 0, 17]].min(0) - 0.15 * mw).astype(int); x1, y1 = (p[[61, 291, 0, 17]].max(0) + 0.15 * mw).astype(int)
    if m.sum() < 30: R.append((k, np.nan, np.nan, np.nan, np.nan)); continue
    ls, lf = cv2.Laplacian(s, cv2.CV_32F), cv2.Laplacian(f, cv2.CV_32F)
    hf = float(lf[m > 0].std() / max(ls[m > 0].std(), 1e-3)); bx = (slice(max(0, y0), max(1, y1)), slice(max(0, x0), max(1, x1)))
    mth = float(lf[bx].std() / max(ls[bx].std(), 1e-3)) if ls[bx].size > 9 else np.nan
    R.append((k, hf, mth, float(np.abs(f - s)[m > 0].mean()), float(np.linalg.norm(p[234] - p[454]))))
R = np.array(R, np.float32); ok = np.isfinite(R[:, 1]); drawn = ok & (R[:, 3] > 2.0); bad = drawn & (R[:, 1] < a.thr)
rng = []; i = 0
while i < len(bad):
    if bad[i]:
        j = i
        while j < len(bad) and (bad[j] or (j + 5 < len(bad) and bad[j:j + 5].any())): j += 1
        if j - i >= 10: rng.append((round(i / 25, 1), round(j / 25, 1), round(float(np.nanmedian(R[i:j, 1])), 2)))
        i = j
    else: i += 1
print("SMEAR %-8s frames %d face in %d drawn %d | face width med %.0f px | hf drawn: med %.2f p10 %.2f | mouth hf med %.2f | smear (hf < %.2f) %d frames = %.0f %% of drawn | ranges %s" % (
    a.name, len(R), int(ok.sum()), int(drawn.sum()), float(np.nanmedian(R[ok, 4])) if ok.any() else 0, float(np.nanmedian(R[drawn, 1])) if drawn.any() else 0, float(np.nanpercentile(R[drawn, 1], 10)) if drawn.any() else 0,
    float(np.nanmedian(R[drawn, 2])) if drawn.any() else 0, a.thr, int(bad.sum()), 100.0 * bad.sum() / max(int(drawn.sum()), 1), rng))
if a.out: np.save(a.out, R)
