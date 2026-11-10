#!/usr/bin/env python3
"""r16c K1 analysis: WHY does the neck <-> face seam jitter? Per frame on a ks_face work dir + its final:
  e      seam excess (Q-style): mean |R_t - R_t-1| in the jaw band (R = final - source, luma) minus the same in a forehead ring (never pasted)
  e_neck the same in the NECK area under the oval mask (generated pixels below the jaw), e_out just outside the mask
drivers:
  gap    generated mouth opening (KeySync landmarks 13-14 / IOD), dgap = |gap_t - gap_t-1|
  chin   generated chin vs source chin, vertical, / IOD (jaw mismatch), dchin = |chin_t - chin_t-1|
  head   source head speed (nose tip, px/frame / IOD)
  ljit   source landmark jitter: chin residual after removing the rigid motion of stable points (t -> t+1), / IOD
Prints Spearman correlations of e with each driver, the e split (band / neck / outside), and the worst seconds with their drivers.
    python seam_cause.py --work W/ks --final FINAL.mp4 [--out C.json]"""
import argparse, json, subprocess, numpy as np, cv2
ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--final", required=True); ap.add_argument("--out", default=""); a = ap.parse_args()
W_ = a.work; JAW = [58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288]
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
STB = [33, 263, 133, 362, 168, 6, 1]
M = f"{W_}/master_25fps.mp4"; L = np.load(f"{W_}/lm_f0.npz")["lm"][..., :2].astype(np.float32)
b = json.load(open(f"{W_}/data/box_f0.json")); bs = b["side"]; xs = b.get("xs"); ys = b.get("ys")
KL = np.load(f"{W_}/post_f0/raw_lm.npz")["lm"][..., :2].astype(np.float32)
W, H = [int(v) for v in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", a.final], capture_output=True, text=True).stdout.strip().split(",")]
def frames(p):
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-i", p, "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    while True:
        bb = pr.stdout.read(W * H)
        if len(bb) < W * H: break
        yield np.frombuffer(bb, np.uint8).astype(np.float32).reshape(H, W)
def bxy(k): return np.array([xs[min(k, len(xs) - 1)] if xs else b["x"], ys[min(k, len(ys) - 1)] if ys else b["y"]], np.float32)
rows = []; pR = None; pp = None
for k, (F, S) in enumerate(zip(frames(a.final), frames(M))):
    R = F - S; p = L[k] if k < len(L) else None
    if p is None or not np.isfinite(p[JAW + STB + [10, 152, 13, 14]]).all(): pR = None; continue
    iod = float(np.linalg.norm(p[33] - p[263]))
    band = np.zeros((H, W), np.uint8); cv2.polylines(band, [np.round(p[JAW]).astype(np.int32)], False, 1, thickness=max(2, int(0.16 * iod)))
    ov = np.zeros((H, W), np.uint8); hp = cv2.convexHull(np.round(p[OVAL]).astype(np.int32))
    cv2.fillPoly(ov, [hp], 1); cv2.fillPoly(ov, [hp + np.array([0, int(0.15 * iod)], np.int32)], 1); ov = cv2.dilate(ov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.2 * iod) | 1,) * 2))
    face = np.zeros((H, W), np.uint8); cv2.fillPoly(face, [hp], 1)
    neck = (ov > 0) & (face == 0); neck[:int(p[152, 1]), :] = False                      # generated, below the jaw
    out = cv2.dilate(ov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.3 * iod) | 1,) * 2)) > 0; out &= ov == 0; out[:int(p[152, 1]), :] = False   # just outside the mask, below the jaw
    fh = np.zeros((H, W), np.uint8); c10 = p[10]; cv2.circle(fh, (int(c10[0]), int(c10[1] - 0.25 * iod)), max(3, int(0.2 * iod)), 1, -1)
    i = min(k, len(KL) - 1); kp = KL[i] * (bs / 512.0) + bxy(k) if np.isfinite(KL[i][[13, 14, 152]]).all() else None
    gap = float(np.linalg.norm(kp[13] - kp[14]) / iod) if kp is not None else np.nan
    chin = float((kp[152, 1] - p[152, 1]) / iod) if kp is not None else np.nan
    if pR is not None and fh.sum() > 20:
        d = np.abs(R - pR); fl = float(d[fh > 0].mean())
        e = float(d[band > 0].mean()) - fl; en = float(d[neck].mean()) - fl if neck.sum() > 30 else np.nan; eo = float(d[out].mean()) - fl if out.sum() > 30 else np.nan
        head = float(np.linalg.norm(p[1] - pp[1]) / iod)
        T, _ = cv2.estimateAffinePartial2D(pp[STB], p[STB], method=cv2.LMEDS)
        lj = float(np.linalg.norm((T[:, :2] @ pp[152] + T[:, 2]) - p[152]) / iod) if T is not None else np.nan
        rows.append([k, e, en, eo, gap, abs(gap - pg) if np.isfinite(gap) and np.isfinite(pg) else np.nan, chin, abs(chin - pc) if np.isfinite(chin) and np.isfinite(pc) else np.nan, head, lj])
    pR, pp, pg, pc = R, p, gap, chin
A = np.array(rows, np.float64); names = ["k", "e", "e_neck", "e_out", "gap", "dgap", "chin", "dchin", "head", "ljit"]
def spear(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20: return np.nan
    rx = np.argsort(np.argsort(x[ok])); ry = np.argsort(np.argsort(y[ok])); return float(np.corrcoef(rx, ry)[0, 1])
res = {"final": a.final, "frames": int(len(A)), "e_median": round(float(np.nanmedian(A[:, 1])), 3), "e_p95": round(float(np.nanpercentile(A[:, 1], 95)), 3),
       "e_neck_median": round(float(np.nanmedian(A[:, 2])), 3), "e_out_median": round(float(np.nanmedian(A[:, 3])), 3),
       "corr": {n: round(spear(A[:, 1], A[:, j]), 3) for j, n in enumerate(names) if j >= 4},
       "drivers_median": {n: round(float(np.nanmedian(A[:, j])), 4) for j, n in enumerate(names) if j >= 4}}
sec = {}
for r in A: sec.setdefault(int(r[0] // 25), []).append(r)
ws = sorted(((float(np.nanmean([x[1] for x in v])), s_, v) for s_, v in sec.items()), key=lambda z: -z[0])[:5]
res["worst"] = [{"s": s_, "e": round(e_, 2), **{n: round(float(np.nanmean([x[j] for x in v])), 3) for j, n in enumerate(names) if j >= 4}} for e_, s_, v in ws]
print("SEAM_CAUSE " + json.dumps(res))
if a.out: json.dump(res, open(a.out, "w"), indent=1); np.save(a.out + ".rows.npy", A)
