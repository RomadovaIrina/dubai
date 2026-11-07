#!/usr/bin/env python3
"""r13 LS: Lab drift of each intermediate LAYER file vs the 25 fps master (OpenCV reader for all), on the background (outside the dilated face:
pixels the generator never changes) and on the lips: which step of the layer chain shifts the colour.
    python scripts/r12/layer_drift.py <id>"""
import sys, subprocess
import numpy as np, cv2, av
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
id_ = sys.argv[1]; W = f"/workspace/dub/artifacts/r4m/r9g_{id_}/work_{id_}_r9g_{id_}"; P = f"{W}/{id_}_r9g_{id_}_25fps"
files = {"ls_raw": P + "_ls_raw.mp4", "ls_pre": P + "_ls_pre.mp4", "cfd": P + "_ls_cfd_ia04.mp4", "srt1": P + "_ls_cfd_ia04_srt1.mp4", "rep2s": P + "_ls_cfd_ia04_srt1_rep2s.mp4"}
lm = np.load(W + "/video/master_mediapipe.npz")["lm"]; n = len(lm); ks = list(range(20, n - 5, max(1, n // 25)))


def rd(p):
    c = cv2.VideoCapture(p); o = {}; k = 0
    while k <= ks[-1]:
        ok, f = c.read()
        if not ok: break
        if k in ks: o[k] = f
        k += 1
    return o


def rd709(p):
    c = av.open(p); o = {}; k = 0
    for fr in c.decode(c.streams.video[0]):
        if k in ks: o[k] = fr.reformat(format="bgr24", src_colorspace="ITU709", src_color_range="MPEG").to_ndarray()
        k += 1
        if k > ks[-1]: break
    return o


def lab(x):
    y = cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32); y[..., 0] *= 100 / 255; y[..., 1:] -= 128; return y


def regions(k, shape):
    p = lm[k]
    if not np.isfinite(p).all(): return None, None
    H, W_ = shape[:2]; iod = np.linalg.norm(p[33, :2] - p[263, :2]); m = np.zeros((H, W_), np.uint8)
    cv2.fillPoly(m, [np.round(cv2.convexHull(p[:, :2].astype(np.float32))[:, 0]).astype(np.int32)], 1)
    bg = ~(cv2.dilate(m, np.ones((int(0.8 * iod) | 1,) * 2, np.uint8)) > 0)
    lp = np.zeros((H, W_), np.uint8); cv2.fillPoly(lp, [np.round(cv2.convexHull(p[LIPS, :2].astype(np.float32))[:, 0]).astype(np.int32)], 1)
    return bg, lp > 0


M = rd(W + "/video/master_25fps.mp4"); MA = rd709(W + "/video/master_25fps.mp4")
print(f"== {id_} master: BT.709 decode minus OpenCV decode dLab", np.round(np.mean([(lab(MA[k]) - lab(M[k])).reshape(-1, 3).mean(0) for k in ks if k in MA and k in M], 0), 2))
for name, p in files.items():
    F = rd(p); d, dl = [], []
    for k in ks:
        if k not in F or k not in M: continue
        bg, lp = regions(k, F[k].shape)
        if bg is None: continue
        d.append((lab(F[k])[bg] - lab(M[k])[bg]).mean(0)); dl.append((lab(F[k])[lp] - lab(M[k])[lp]).mean(0))
    tag = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=color_space", "-of", "csv=p=0", p], capture_output=True, text=True).stdout.strip()
    print(f"== {id_} {name:6s} bg dLab {np.round(np.mean(d, 0), 2)}  lips dLab {np.round(np.mean(dl, 0), 2)}  tag {tag}", flush=True)
