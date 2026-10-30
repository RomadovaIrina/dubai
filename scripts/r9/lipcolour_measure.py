#!/usr/bin/env python3
"""r9: lip colour vs the source ("lips look over-saturated red, lipstick-like" on m18 r9fx). Inside the lip RING (outer minus inner lip polygon,
eroded 1 px), each file with its OWN MediaPipe landmarks, on the 25 fps master timeline: mean Lab L, a* (red), b* (yellow), chroma; reported
as the difference to the source on the same frames. Layers: raw LatentSync, + CodeFormer, + detail / SR, and the final outputs of the arms.
    python lipcolour.py <id> <tag> ...   (final outputs are resampled to 25 fps)"""
import pathlib, subprocess, sys, json
import numpy as np, cv2
sys.path.insert(0, "/workspace/dub/dubai/scripts/pilot"); sys.path.insert(0, "/workspace/dub/dubai/scripts/r8")
import clean_face_stack as cfs
from r8_mouth_flicker import LIPS_INNER, LIPS_OUTER
B = pathlib.Path("/workspace/dub/artifacts/r4m"); D = pathlib.Path("/workspace/dub/r9/diag")


def lab_stats(video, lm, n):
    c = cv2.VideoCapture(str(video)); out = np.full((n, 4), np.nan)
    for k in range(n):
        ok, f = c.read()
        if not ok: break
        p = lm[k]
        if not np.isfinite(p[LIPS_OUTER + LIPS_INNER]).all(): continue
        m = np.zeros(f.shape[:2], np.uint8); cv2.fillPoly(m, [np.round(p[LIPS_OUTER, :2]).astype(np.int32)], 1); cv2.fillPoly(m, [np.round(p[LIPS_INNER, :2]).astype(np.int32)], 0)
        m = cv2.erode(m, np.ones((3, 3), np.uint8))
        if m.sum() < 20: continue
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)[m > 0]; L, a, b = lab[:, 0] * 100 / 255, lab[:, 1] - 128, lab[:, 2] - 128
        out[k] = (L.mean(), a.mean(), b.mean(), np.hypot(a, b).mean())
    return out


i = sys.argv[1]; W = B / f"r9g_{i}" / f"work_{i}_r9g_{i}"; st = f"{i}_r9g_{i}_25fps"
src = lab_stats(W / "video" / "master_25fps.mp4", np.load(W / "video" / "master_mediapipe.npz")["lm"], 10 ** 6 if False else len(np.load(W / "video" / "master_mediapipe.npz")["lm"]))
n = len(src); files = [("raw LatentSync", W / f"{st}_ls_pre.mp4"), ("+ CodeFormer", W / f"{st}_ls_raw.mp4"), ("+ detail + SR", W / f"{st}_ls_cfd_cfd_srt1.mp4"), ("+ detail 0.4 + SR", W / f"{st}_ls_cfd_ia04_srt1.mp4")]
for T in sys.argv[2:]:
    o25 = D / f"{T}_25.mp4"
    if not o25.exists(): subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(B / T / f"{i}_{T}.mp4"), "-r", "25", "-an", "-c:v", "libx264", "-crf", "12", str(o25)], check=True)
    files.append((f"final {T}", o25))
res = {"source": dict(zip(("L", "a", "b", "chroma"), np.round(np.nanmean(src, 0), 2).tolist()))}
print(f"{i} source lips: L {res['source']['L']} a* {res['source']['a']} b* {res['source']['b']} chroma {res['source']['chroma']}")
for lab, p in files:
    if not p.exists(): continue
    cache = p.with_suffix(".lipc_mp.npz") if "final" not in lab else D / f"{p.stem}_lipc_mp.npz"
    try: lm = np.load(cfs.run_mediapipe(p, cache))["lm"]
    except RuntimeError: lm = np.load(cfs.run_mediapipe(p, cache.with_name(cache.stem + "_img.npz"), mode="image"))["lm"]   # broken fps tag (old cf_detail bug)
    s = lab_stats(p, lm, n); ok = np.isfinite(s[:, 0]) & np.isfinite(src[:, 0]); d = np.nanmean(s[ok] - src[ok], 0)
    res[lab] = dict(zip(("dL", "da", "db", "dchroma"), np.round(d, 2).tolist())); print(f"  {lab:28s} dL {d[0]:+.2f}  da* {d[1]:+.2f}  db* {d[2]:+.2f}  dchroma {d[3]:+.2f}  (frames {ok.sum()})")
(D / f"lipcolour_{i}.json").write_text(json.dumps(res, indent=1))
