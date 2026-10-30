#!/usr/bin/env python3
"""r9: "the teeth look drawn". Inside the inner-lip polygon on open-mouth frames (o > 0.04), per layer vs the source (25 fps master):
  hf    = mean |Laplacian| of luma (fine detail / drawn edges), as a ratio to the source's on the same frames
  edge  = 95th-percentile Sobel magnitude (crisp outlines), ratio to the source
  lum90 = 90th-percentile luma (how bright the teeth are), difference to the source
Uses the landmark caches written by artefacts.py (<layer>.art_mp.npz; the output's <F>_<stem>_mp.npz).   python teeth_hf.py <id> [suffix]"""
import pathlib, sys, json
import numpy as np, cv2
sys.path.insert(0, "/workspace/dub/dubai/scripts/r8")
from r8_mouth_flicker import LIPS_INNER
i = sys.argv[1]; sx = sys.argv[2] if len(sys.argv) > 2 else ""; B = pathlib.Path("/workspace/dub/artifacts/r4m"); D = pathlib.Path("/workspace/dub/r9/diag"); GEN = sys.argv[3] if len(sys.argv) > 3 else ""; L3 = sys.argv[4] if len(sys.argv) > 4 else ""
G, F = (GEN or f"r9g_{i}{sx}"), f"r9f_{i}{sx}"; W = B / G / f"work_{i}_{G}"; st = f"{i}_{G}_25fps"; W0 = B / f"r9g_{i}" / f"work_{i}_r9g_{i}"
src = (W0 / "video" / "master_25fps.mp4", np.load(W0 / "video" / "master_mediapipe.npz")["lm"])
L = [("L1 raw LatentSync", W / f"{st}_ls_pre.mp4", W / f"{st}_ls_pre.art_mp.npz"), ("L2 +CodeFormer", W / f"{st}_ls_raw.mp4", W / f"{st}_ls_raw.art_mp.npz"),
     ("L3 +detail+SR", pathlib.Path(L3) if L3 else W / f"{st}_ls_cfd_cfd_srt1.mp4", pathlib.Path(L3).with_suffix(".art_mp.npz") if L3 else W / f"{st}_ls_cfd_cfd_srt1.art_mp.npz"), ("L4 final", D / f"{F}_25.mp4", D / f"{F}_{F}_25_mp.npz")]
lsm = np.load(W / f"{st}_fb_fb7.npz")["ls_mask"].astype(bool)


def stats(video, lm, frames):
    c = cv2.VideoCapture(str(video)); out = {}; k = 0
    while True:
        ok, f = c.read()
        if not ok or k >= len(lm): break
        if k in frames:
            p = lm[k]
            if np.isfinite(p[LIPS_INNER]).all():
                m = np.zeros(f.shape[:2], np.uint8); cv2.fillPoly(m, [np.round(p[LIPS_INNER, :2]).astype(np.int32)], 1); m = cv2.erode(m, np.ones((3, 3), np.uint8))
                if m.sum() >= 30:
                    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32); lap = np.abs(cv2.Laplacian(g, cv2.CV_32F)); sob = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
                    out[k] = (float(lap[m > 0].mean()), float(np.percentile(sob[m > 0], 95)), float(np.percentile(g[m > 0], 90)))
        k += 1
    return out


so = np.linalg.norm(src[1][:, 13, :2] - src[1][:, 14, :2], axis=1) / np.linalg.norm(src[1][:, 10, :2] - src[1][:, 152, :2], axis=1)
frames = set(np.nonzero(lsm[:len(so)] & (so > 0.04))[0].tolist()); S = stats(src[0], src[1], frames); res = {"id": i, "arm": sx or "r9", "open_frames": len(S)}
for name, v, lmp in L:
    if not (v.exists() and lmp.exists()): res[name] = "missing"; continue
    X = stats(v, np.load(lmp)["lm"], frames); ks = [k for k in X if k in S]
    hf = np.median([X[k][0] / max(1e-3, S[k][0]) for k in ks]); ed = np.median([X[k][1] / max(1e-3, S[k][1]) for k in ks]); lu = np.median([X[k][2] - S[k][2] for k in ks])
    res[name] = {"hf_ratio": round(float(hf), 2), "edge_ratio": round(float(ed), 2), "lum90_minus_src": round(float(lu), 1), "frames": len(ks)}
print(json.dumps(res)); (D / f"teeth_{i}{sx}.json").write_text(json.dumps(res, indent=1))
