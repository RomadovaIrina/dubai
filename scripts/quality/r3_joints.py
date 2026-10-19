#!/usr/bin/env python3
"""Round 3 track B: LatentSync window-joint visibility on an OUTPUT video (any fps). For each pair of consecutive output frames inside
LatentSync segments: motion-compensated residual of the mouth / lower-face ROI using the OUTPUT's own flow (k -> k-1), total and
low-frequency part (Gaussian 0.15 mouth width). A pair is a joint pair when a 16-frame window start lies between the two frames' master
positions. Reports joint / inner ratios (1.0 = joints invisible) and the share of joints above 1.5x the local inner median.
    python scripts/quality/r3_joints.py --video OUT --manifest MAN --lm timing/<id>_source_lm.npz [--ref-master timing.json] --out x.json"""
import argparse, json, pathlib
import numpy as np, cv2, av
ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--lm", required=True)
ap.add_argument("--ref-master", default=None, help="the video is on the master grid (25 fps): frame i = master i, landmarks lm[j[i]]"); ap.add_argument("--out", required=True)
a = ap.parse_args()
z = np.load(a.lm); lm = z["lm"]; man = json.loads(pathlib.Path(a.manifest).read_text())
if a.ref_master:
    j = np.array(json.loads(pathlib.Path(a.ref_master).read_text())["j"]); lm = lm[np.clip(j, 0, len(lm) - 1)]; mpos = np.arange(len(lm), dtype=float)
else: mpos = z["t"] * 25.0
segs = [s for s in man["lipsync"]["segments"] if s["action"] == "LATENT_SYNC"]
def seg_of(m):
    for s in segs:
        if s["start_frame"] <= m < s["end_frame"]: return s
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); rows = []; prev = None
for k, fr in enumerate(av.open(a.video).decode(video=0)):
    if k >= len(lm): break
    f = fr.to_ndarray(format="bgr24")
    if prev is not None and np.isfinite(lm[k]).all():
        s0, s1 = seg_of(mpos[k - 1]), seg_of(mpos[k])
        if s0 is not None and s0 is s1 and mpos[k - 1] >= s0["start_frame"] + 2:
            p = lm[k]; c = p[[61, 291, 13, 14]].mean(0); w = float(np.linalg.norm(p[61] - p[291]))
            x0, x1, y0, y1 = int(c[0] - 1.1 * w), int(c[0] + 1.1 * w), int(c[1] - 0.8 * w), int(c[1] + 1.1 * w)
            A = prev[y0:y1, x0:x1].astype(np.float32); B = f[y0:y1, x0:x1].astype(np.float32)
            if A.size and A.shape == B.shape:
                F = dis.calc(cv2.cvtColor(B.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.cvtColor(A.astype(np.uint8), cv2.COLOR_BGR2GRAY), None)
                yy, xx = np.mgrid[0:B.shape[0], 0:B.shape[1]].astype(np.float32)
                r = B - cv2.remap(A, xx + F[..., 0], yy + F[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                b0 = s0["start_frame"]; joint = np.floor((mpos[k] - b0) / 16) > np.floor((mpos[k - 1] - b0) / 16)
                rows.append((k, joint, float(np.abs(r).mean()), float(np.abs(cv2.GaussianBlur(r, (0, 0), 0.15 * w)).mean())))
    prev = f
R = np.array(rows, float); jt = R[:, 1] == 1; out = {"video": a.video, "pairs": len(R), "joints": int(jt.sum())}
for i, nm in ((2, "resid"), (3, "lowfreq")):
    loc = []
    for q in np.flatnonzero(jt):
        nb = [x for x in range(q - 8, q + 9) if 0 <= x < len(R) and not jt[x] and abs(R[x, 0] - R[q, 0]) <= 10]
        if nb: loc.append(R[q, i] / np.median(R[nb, i]))
    loc = np.array(loc)
    out[nm] = {"joint_over_inner": round(float(R[jt, i].mean() / R[~jt, i].mean()), 3), "local_median": round(float(np.median(loc)), 3), "share_gt_1_5": round(float((loc > 1.5).mean()), 3)}
pathlib.Path(a.out).write_text(json.dumps(out, indent=1)); print(json.dumps(out))
