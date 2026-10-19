#!/usr/bin/env python3
"""Round 3 D: single-frame "flash" detector of the mouth region (torn / whitish lips on one output frame, e.g. a bad 25->30 fps interpolation).
Per output frame k, in a box around the SOURCE frame's lips (MediaPipe on the source, the output keeps the source geometry):
  hf = mean |Laplacian| (texture / tearing), lum = mean grey level (whitish lips), col = mean |a|+|b| chroma (Lab).
spike_k(x) = x_k - median(x_{k-2}, x_{k-1}, x_{k+1}, x_{k+2}), normalised by the video's robust scale of x (1.4826 MAD of first differences
of the SOURCE series). A frame flashes when max(|spike| hf, lum, col) > --thr in the output while the source spike at k is < thr / 2.
    python scripts/quality/r3_flash.py --video OUT.mp4 --src SRC.mp4 [--lm SRC_lm.npz] [--out x.json] [--frames a:b]"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


def decode(p, a=0, b=None):
    import av
    c = av.open(str(p)); out = []
    for i, f in enumerate(c.decode(c.streams.video[0])):
        if b is not None and i >= b: break
        if i >= a: out.append(f.to_ndarray(format="bgr24"))
    c.close(); return out


def series(frames, lm, a):
    import cv2
    hf, lum, col = [], [], []
    for i, f in enumerate(frames):
        p = lm[a + i] if a + i < len(lm) else None
        if p is None or not np.isfinite(p[LIPS]).all(): hf.append(np.nan); lum.append(np.nan); col.append(np.nan); continue
        q = p[LIPS, :2]; w = q[:, 0].max() - q[:, 0].min(); x0, y0 = q.min(0) - 0.15 * w; x1, y1 = q.max(0) + 0.15 * w
        x0, y0 = max(0, int(x0)), max(0, int(y0)); x1, y1 = min(f.shape[1], int(x1) + 1), min(f.shape[0], int(y1) + 1)
        if x1 - x0 < 6 or y1 - y0 < 6: hf.append(np.nan); lum.append(np.nan); col.append(np.nan); continue
        crop = f[y0:y1, x0:x1]; g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32); lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        hf.append(float(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)).mean())); lum.append(float(g.mean()))
        col.append(float(np.abs(lab[..., 1] - 128).mean() + np.abs(lab[..., 2] - 128).mean()))
    return {"hf": np.array(hf), "lum": np.array(lum), "col": np.array(col)}


def mc_resid(frames, lm, a):
    """motion-compensated residual of the lip box: frame k vs its best-matching neighbour (k-1 or k+1) warped onto it by DIS flow.
    A ghosted / torn frame (two half-matching mouths averaged) matches neither neighbour."""
    import cv2
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); out = np.full(len(frames), np.nan)
    for i in range(1, len(frames) - 1):
        p = lm[a + i] if a + i < len(lm) else None
        if p is None or not np.isfinite(p[LIPS]).all(): continue
        q = p[LIPS, :2]; w = q[:, 0].max() - q[:, 0].min(); x0, y0 = (q.min(0) - 0.25 * w).astype(int); x1, y1 = (q.max(0) + 0.25 * w).astype(int)
        x0, y0 = max(0, x0), max(0, y0)
        if x1 - x0 < 8 or y1 - y0 < 8: continue
        g = [cv2.cvtColor(frames[j][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY) for j in (i - 1, i, i + 1)]
        if any(x.shape != g[1].shape for x in g): continue
        r = []
        for nb in (g[0], g[2]):
            F = dis.calc(g[1], nb, None); h, ww = F.shape[:2]; gx, gy = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(h, dtype=np.float32))
            wp = cv2.remap(nb, gx + F[..., 0], gy + F[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); r.append(float(np.abs(wp.astype(np.float32) - g[1]).mean()))
        out[i] = min(r)
    return out


def spikes(x):
    n = len(x); s = np.full(n, np.nan)
    for k in range(2, n - 2):
        nb = x[[k - 2, k - 1, k + 1, k + 2]]
        if np.isfinite(x[k]) and np.isfinite(nb).all(): s[k] = x[k] - np.median(nb)
    return s


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--src", required=True); ap.add_argument("--lm", default=None)
    ap.add_argument("--out", default=None); ap.add_argument("--frames", default=None); ap.add_argument("--thr", type=float, default=6.0)
    a = ap.parse_args()
    import clean_face_stack as cfs
    lmp = a.lm or str(cfs.run_mediapipe(pathlib.Path(a.src), pathlib.Path("/workspace/dub/artifacts/r3d/timing") / (pathlib.Path(a.src).stem + "_src_mediapipe.npz")))
    lm = np.load(lmp)["lm"]
    fa, fb = (int(x) for x in a.frames.split(":")) if a.frames else (0, None)
    V, S = decode(a.video, fa, fb), decode(a.src, fa, fb); n = min(len(V), len(S))
    sv, ss = series(V[:n], lm, fa), series(S[:n], lm, fa)
    z = {}; flags = np.zeros(n, bool); src_flags = np.zeros(n, bool)
    for key in ("hf", "lum", "col"):
        d = np.diff(ss[key]); d = d[np.isfinite(d)]; scale = max(1.4826 * np.median(np.abs(d - np.median(d))) if len(d) else 1.0, 1e-3)
        zv, zs = spikes(sv[key]) / scale, spikes(ss[key]) / scale; z[key] = (zv, zs)
        flags |= np.nan_to_num(np.abs(zv)) > a.thr; src_flags |= np.nan_to_num(np.abs(zs)) > a.thr
    zmax_v = np.nanmax(np.stack([np.abs(z[k][0]) for k in z]), 0); zmax_s = np.nanmax(np.stack([np.abs(z[k][1]) for k in z]), 0)
    fl_stat = flags & ~(np.nan_to_num(zmax_s) > a.thr / 2)
    # structural criterion (the observed defect): residual >= 1.5x its neighbours' median AND >= 1.6x the source's at the same frame
    mv, ms = mc_resid(V[:n], lm, fa), mc_resid(S[:n], lm, fa); fl_mc = np.zeros(n, bool); ratio = np.full(n, np.nan)
    for k in range(2, n - 2):
        nb = mv[[k - 2, k - 1, k + 1, k + 2]]
        if np.isfinite(mv[k]) and np.isfinite(nb).all() and np.isfinite(ms[k]):
            ratio[k] = mv[k] / max(np.median(nb), 1e-3); fl_mc[k] = ratio[k] >= 1.5 and mv[k] >= 1.6 * max(ms[k], 0.5)
    fl = fl_mc | fl_stat
    idx = np.flatnonzero(fl) + fa
    r = {"video": a.video, "frames": n, "thr": a.thr, "flash_frames": int(fl.sum()), "source_spike_frames": int(src_flags.sum()),
         "flash_mc": int(fl_mc.sum()), "flash_stat": int(fl_stat.sum()),
         "flash_list": [{"frame": int(k), "t": None, "mc_ratio": None if not np.isfinite(ratio[k - fa]) else round(float(ratio[k - fa]), 2),
                         "mc_vs_src": round(float(mv[k - fa] / max(ms[k - fa], 0.5)), 2) if np.isfinite(mv[k - fa]) and np.isfinite(ms[k - fa]) else None, "z_hf": round(float(z["hf"][0][k - fa]), 1), "z_lum": round(float(z["lum"][0][k - fa]), 1),
                         "z_col": round(float(z["col"][0][k - fa]), 1)} for k in idx[:200]]}
    try:
        import av
        c = av.open(a.video); s = c.streams.video[0]; ts = [float(f.pts * s.time_base) for f in c.decode(s)]; c.close()
        for e in r["flash_list"]: e["t"] = round(ts[e["frame"]], 3) if e["frame"] < len(ts) else None
    except Exception: pass
    print(json.dumps({k: r[k] for k in ("video", "frames", "flash_frames", "flash_mc", "flash_stat", "source_spike_frames")} | {"first": r["flash_list"][:12]}))
    if a.out: pathlib.Path(a.out).write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
