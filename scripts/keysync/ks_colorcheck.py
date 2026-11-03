#!/usr/bin/env python3
"""r13 KS quality: colour jumps + sharpness of a pasted / generated face vs its reference, per frame.
A (result) and B (reference) are read at the same size and frame rate (B is resampled to A's fps). Inside the box (or the full frame), with the
face oval from a mediapipe npz of A (absolute pixel coords, same frames as A), the regions are: up = the oval above the nose tip (KeySync
should not change it), low = the oval below, bg = the box outside the dilated oval. Per frame and region: Lab mean / std of A - B, and the
Laplacian variance ratio A / B (sharpness). A colour JUMP is a step in the A - B mean: |mean(d[t:t+w]) - mean(d[t-w:t])| in dE (Lab) > --thr,
non-max suppressed over 2w frames.
    ks_colorcheck.py --a A.mp4 --b B.mp4 [--box box.json] [--lm A.mediapipe.npz] [--shift S] --out OUT.json"""
import argparse, json, subprocess, numpy as np, cv2
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--box", default="")
ap.add_argument("--lm", default=""); ap.add_argument("--shift", type=int, default=0, help="A frame i is compared with B frame i - shift")
ap.add_argument("--w", type=int, default=6); ap.add_argument("--thr", type=float, default=2.0); ap.add_argument("--out", required=True); a = ap.parse_args()
pr = lambda f, e: subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", f"stream={e}", "-of", "csv=p=0", f], capture_output=True, text=True).stdout.strip()
W, H = [int(v) for v in pr(a.a, "width,height").split(",")]; fps = pr(a.a, "r_frame_rate"); fr = eval(fps)
def rd(p, vf):
    q = subprocess.Popen(["ffmpeg", "-v", "error", "-i", p, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, bufsize=W * H * 12)
    while True:
        r = q.stdout.read(W * H * 3)
        if len(r) < W * H * 3: return
        yield np.frombuffer(r, np.uint8).reshape(H, W, 3)
if a.box: b = json.load(open(a.box)); x0, y0, s = b["x"], b["y"], b["side"]; bw = bh = s
else: x0, y0, bw, bh = 0, 0, W, H
LM = np.load(a.lm)["lm"][..., :2].astype(np.float32) if a.lm else None
Bs = list(rd(a.b, f"fps={fps},scale={W}:{H}"))
R = {k: [] for k in ["t", "dL", "da", "db", "dsL", "sharp"]}; reg = ["up", "low", "bg"]
lab = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
lv = lambda g, m: cv2.Laplacian(g, cv2.CV_32F)[m].var() if m.sum() > 50 else np.nan
for i, fa in enumerate(rd(a.a, "null")):
    j = i - a.shift
    if j < 0 or j >= len(Bs): continue
    A_, B_ = fa[y0:y0 + bh, x0:x0 + bw], Bs[j][y0:y0 + bh, x0:x0 + bw]
    m = {}
    if LM is not None and i < len(LM) and np.isfinite(LM[i]).all() and LM[i].max() > 0:
        p = LM[i]; ov = np.zeros((bh, bw), np.uint8); cv2.fillPoly(ov, [np.round(p[OVAL] - [x0, y0]).astype(np.int32)], 1)
        ny = int(p[1, 1] - y0); rows = np.arange(bh)[:, None]; fw = int(np.ptp(p[OVAL, 0]))
        m["up"] = (ov > 0) & (rows < ny); m["low"] = (ov > 0) & (rows >= ny)
        m["bg"] = cv2.dilate(ov, np.ones((3, 3), np.uint8), iterations=max(1, fw // 8)) == 0
    else:
        m = {"up": np.zeros((bh, bw), bool), "low": np.zeros((bh, bw), bool), "bg": np.ones((bh, bw), bool)}; m["up"][: bh // 3, bw // 4: 3 * bw // 4] = True
    La, Lb = lab(A_), lab(B_); ga, gb = cv2.cvtColor(A_, cv2.COLOR_BGR2GRAY), cv2.cvtColor(B_, cv2.COLOR_BGR2GRAY)
    R["t"].append(i / fr)
    for k, ch in [("dL", 0), ("da", 1), ("db", 2)]: R[k].append([float(La[m[r]][:, ch].mean() - Lb[m[r]][:, ch].mean()) if m[r].sum() > 50 else np.nan for r in reg])
    R["dsL"].append([float(La[m[r]][:, 0].std() - Lb[m[r]][:, 0].std()) if m[r].sum() > 50 else np.nan for r in reg])
    R["sharp"].append([float(lv(ga, m[r]) / max(lv(gb, m[r]), 1e-3)) for r in reg])
t = np.array(R["t"]); D = np.stack([np.array(R[k]) for k in ["dL", "da", "db"]], -1)   # (n, region, Lab)
D = np.where(np.isfinite(D), D, np.nan); n = len(t); w = a.w
jumps = {}
for ri, r in enumerate(reg):
    d = D[:, ri]; st = np.zeros(n)
    for i in range(w, n - w):
        st[i] = np.linalg.norm(np.nanmean(d[i:i + w], 0) - np.nanmean(d[i - w:i], 0))
    st = np.nan_to_num(st); pk = []
    for i in np.argsort(-st):
        if st[i] < a.thr: break
        if all(abs(i - q) > 2 * w for q in pk): pk.append(int(i))
    jumps[r] = sorted([{"t": round(float(t[i]), 2), "frame": i, "dE": round(float(st[i]), 2), "dLab": np.round(np.nanmean(d[i:i + w], 0) - np.nanmean(d[i - w:i], 0), 2).tolist()} for i in pk], key=lambda q: q["frame"])
S = np.array(R["sharp"])
summ = {"frames": n, "fps": fr, "mean_dLab": {r: np.round(np.nanmean(D[:, ri], 0), 2).tolist() for ri, r in enumerate(reg)},
        "std_dLab_over_time": {r: np.round(np.nanstd(D[:, ri], 0), 2).tolist() for ri, r in enumerate(reg)},
        "sharp_ratio_median": {r: round(float(np.nanmedian(S[:, ri])), 3) for ri, r in enumerate(reg)}, "jumps": jumps}
json.dump({**summ, "per_frame": {k: np.round(np.array(v, dtype=float), 3).tolist() for k, v in R.items()}}, open(a.out, "w"))
print(json.dumps({k: v for k, v in summ.items() if k != "jumps"})); [print(f"JUMPS {r} {len(v)}: " + " ".join(f"{q['t']}s/{q['dE']}" for q in v)) for r, v in jumps.items()]
