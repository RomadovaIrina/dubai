#!/usr/bin/env python3
"""Round 3 D: count generator "blob" frames in a raw LatentSync layer (washed-out lips: the lip-box Laplacian or contrast collapses vs the
frames 2..6 away; same rule as fps_composite --sanitize-layer).  python scripts/quality/r3_blobcount.py LS_RAW.mp4 MANIFEST [--thr 0.83]"""
import argparse, json, pathlib, sys
import numpy as np, cv2
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parent), str(pathlib.Path(__file__).resolve().parents[1] / "pilot")]
import temporal_mouth_stabilize as tms, clean_face_stack as cfs
from fps_composite import master_weights
ap = argparse.ArgumentParser(); ap.add_argument("ls_raw"); ap.add_argument("manifest"); ap.add_argument("--thr", type=float, default=0.83); a = ap.parse_args()
man = json.loads(pathlib.Path(a.manifest).read_text()); L = tms.read_video(a.ls_raw); nm = len(L); wm = master_weights(man, nm)
mp = np.load(cfs.run_mediapipe(pathlib.Path(a.ls_raw), pathlib.Path(a.ls_raw).with_suffix(".mediapipe.npz")))["lm"][:nm]
hf = np.full(nm, np.nan); ct = np.full(nm, np.nan)
for m in range(nm):
    if wm[m] <= 0 or not np.isfinite(mp[m][tms.LIPS_OUTER]).all(): continue
    q = mp[m][tms.LIPS_OUTER, :2]; w = q[:, 0].max() - q[:, 0].min(); x0, y0 = (q.min(0) - 0.1 * w).astype(int); x1, y1 = (q.max(0) + 0.1 * w).astype(int)
    g = cv2.cvtColor(L[m][max(0, y0):y1, max(0, x0):x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    if g.size >= 50: hf[m] = np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)).mean(); ct[m] = g.std()
bad = []
for m in range(nm):
    if not np.isfinite(hf[m]): continue
    nb = [i for i in range(max(0, m - 6), min(nm, m + 7)) if abs(i - m) >= 2 and np.isfinite(hf[i])]
    if len(nb) >= 4 and min(hf[m] / np.median(hf[nb]), ct[m] / np.median(ct[nb])) < a.thr: bad.append(m)
ls = int(np.isfinite(hf).sum())
print(json.dumps({"ls_raw": a.ls_raw, "ls_frames": ls, "blob_frames": len(bad), "share": round(len(bad) / max(ls, 1), 4), "first": bad[:40]}))
