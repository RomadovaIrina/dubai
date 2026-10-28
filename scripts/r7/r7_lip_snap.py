#!/usr/bin/env python3
"""r7 F (coordinator: 'lip snaps'): LatentSync sometimes jumps the mouth between two frames (01 m8 7.5 s: opening 0.128 -> 0.023 IOD in one
frame) or draws one odd mouth inside a closure (01 m8 frame 190: a pursed bulge between two closed frames; 04 frame 985). Both are a spike
of the lips-shape SECOND difference on the layer's own landmarks: s[k] = |p[k-1] - 2 p[k] + p[k+1]| over the lips points (IOD units, head
motion removed by centring on the lips). Frames with s[k] > --thr are re-drawn as the flow interpolation of k-1 and k+1 (midpoint, DIS
flow both ways on the mouth crop, confidence-weighted like fps_composite.interp_safe), inside a feathered lips-hull mask; the landmarks of
those frames become the neighbours' mean. A genuine one-frame bilabial (closed frame between two OPEN frames: opening < --dip-close with
both neighbours > --dip-open) is kept, so the mouth still closes on p/b/m; only runs of <= --max-run flagged frames are touched.
    python scripts/r7/r7_lip_snap.py --ls-raw X_ls_raw.mp4 --lm X_ls_raw.mediapipe.npz --out X_ls_snap.mp4 [--thr 0.045] [--dry]
"""
from __future__ import annotations
import argparse, json, pathlib, time
import numpy as np

LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
LIPS = LIPS_OUTER + LIPS_INNER


def shape(lm):
    """per frame: lips points centred on their mean, in IOD units (nan where no face); the mouth opening (13-14 / IOD)"""
    n = len(lm); S = np.full((n, len(LIPS), 2), np.nan, np.float32); O = np.full(n, np.nan, np.float32)
    for k in range(n):
        p = lm[k]
        if p is None or not np.isfinite(p[LIPS + [33, 263]]).all(): continue
        iod = float(np.linalg.norm(p[33, :2] - p[263, :2]));
        if iod < 1: continue
        q = p[LIPS, :2] / iod; S[k] = q - q.mean(0); O[k] = float(np.linalg.norm(p[13, :2] - p[14, :2])) / iod
    return S, O


def detect(S, O, thr, dip_open, dip_close, max_run):
    n = len(S); s = np.full(n, np.nan, np.float32)
    for k in range(1, n - 1):
        if np.isfinite(S[k - 1:k + 2]).all(): s[k] = float(np.linalg.norm(S[k - 1] - 2 * S[k] + S[k + 1], axis=1).mean())
    flag = np.isfinite(s) & (s > thr)
    kept_dip = [k for k in np.flatnonzero(flag) if O[k] < dip_close and O[k - 1] > dip_open and O[k + 1] > dip_open]
    flag[kept_dip] = False
    runs = []; k = 0
    while k < n:
        if flag[k]:
            j = k
            while j + 1 < n and flag[j + 1]: j += 1
            runs.append((k, j)); k = j + 1
        else: k += 1
    too_long = [r for r in runs if r[1] - r[0] + 1 > max_run]
    for r in too_long: flag[r[0]:r[1] + 1] = False
    return s, flag, kept_dip, too_long


def main():
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ls-raw", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.045, help="lips-shape second difference (IOD units, mean over the 40 lips points)")
    ap.add_argument("--dip-open", type=float, default=0.08); ap.add_argument("--dip-close", type=float, default=0.03); ap.add_argument("--max-run", type=int, default=2)
    ap.add_argument("--dilate", type=float, default=0.6); ap.add_argument("--feather", type=float, default=0.2); ap.add_argument("--crf", type=int, default=10)
    ap.add_argument("--dry", action="store_true", help="detect only, print the flagged frames")
    a = ap.parse_args(); lm = np.load(a.lm)["lm"]; S, O = shape(lm); s, flag, kept_dip, too_long = detect(S, O, a.thr, a.dip_open, a.dip_close, a.max_run)
    fr = [int(k) for k in np.flatnonzero(flag)]
    rep = {"thr": a.thr, "flagged": len(fr), "frames": fr, "kept_dip": [int(k) for k in kept_dip], "runs_too_long": [[int(x) for x in r] for r in too_long],
           "s_p99": float(np.nanpercentile(s, 99)) if np.isfinite(s).any() else None, "s_max": float(np.nanmax(s)) if np.isfinite(s).any() else None,
           "scores": {int(k): round(float(s[k]), 4) for k in fr}}
    print(f"== lip_snap {pathlib.Path(a.ls_raw).name}: {len(fr)} flagged {fr[:40]} kept_dip {rep['kept_dip']} too_long {rep['runs_too_long']} s_p99 {rep['s_p99']}")
    if a.dry: return
    import av
    t0 = time.perf_counter(); c = av.open(a.ls_raw); st_in = c.streams.video[0]; frames = [f.to_ndarray(format="bgr24") for f in c.decode(st_in)]; c.close()
    n = len(frames); H, W = frames[0].shape[:2]; out = [f for f in frames]; lm2 = lm.copy()
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); gx, gy = None, None
    def warp(img, F):
        h, w = F.shape[:2]; gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        return cv2.remap(img, gx + F[..., 0], gy + F[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    done = 0; lowconf = 0
    for k in fr:
        if k < 1 or k + 1 >= n: continue
        p = lm[k]; q = p[LIPS_OUTER, :2]; mw = float(q[:, 0].max() - q[:, 0].min()); pad = (a.dilate + a.feather + 0.3) * mw
        x0, y0 = int(max(0, q[:, 0].min() - pad)), int(max(0, q[:, 1].min() - pad)); x1, y1 = int(min(W, q[:, 0].max() + pad)), int(min(H, q[:, 1].max() + pad))
        if x1 - x0 < 16 or y1 - y0 < 16: continue
        y, z = frames[k - 1][y0:y1, x0:x1].astype(np.float32), frames[k + 1][y0:y1, x0:x1].astype(np.float32)   # the ORIGINAL neighbours, not re-drawn ones
        gy_, gz_ = cv2.cvtColor(np.clip(y, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.cvtColor(np.clip(z, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        F01, F10 = dis.calc(gy_, gz_, None), dis.calc(gz_, gy_, None); t = 0.5
        Ft0 = -(1 - t) * t * F01 + t * t * F10; Ft1 = (1 - t) ** 2 * F01 - t * (1 - t) * F10
        w0, w1 = warp(y, Ft0), warp(z, Ft1); it = 0.5 * (w0 + w1)
        dis_ = cv2.GaussianBlur(np.abs(w0 - w1).mean(2), (0, 0), 2.0)
        if float(np.median(dis_)) > 12: lowconf += 1   # the two warps disagree: the nearer-looking single frame avoids a double contour
        wr = np.clip((dis_ - 6.0) / 6.0, 0, 1)[..., None]; near = w0 if np.abs(w0 - frames[k][y0:y1, x0:x1]).mean() < np.abs(w1 - frames[k][y0:y1, x0:x1]).mean() else w1
        it = it + wr * (near - it)
        ch, cw = it.shape[:2]; hull = cv2.convexHull((q - [x0, y0]).astype(np.float32))[:, 0]; m = np.zeros((ch, cw), np.float32)
        cv2.fillPoly(m, [np.round(hull).astype(np.int32)], 1.0); r = int(max(1, a.dilate * mw))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))); m = cv2.GaussianBlur(m, (0, 0), max(0.5, a.feather * mw / 2.563))[..., None]
        f = frames[k].copy(); f[y0:y1, x0:x1] = np.clip(frames[k][y0:y1, x0:x1] * (1 - m) + it * m, 0, 255).round().astype(np.uint8); out[k] = f
        lm2[k] = 0.5 * (lm[k - 1] + lm[k + 1]); done += 1
    o = av.open(a.out, "w"); st = o.add_stream("libx264", rate=st_in.average_rate); st.width, st.height, st.pix_fmt = W, H, "yuv420p"; st.options = {"crf": str(a.crf), "preset": "medium"}
    for f in out:
        for pk in st.encode(av.VideoFrame.from_ndarray(f, format="bgr24")): o.mux(pk)
    for pk in st.encode(): o.mux(pk)
    o.close(); src_npz = np.load(a.lm, allow_pickle=True); np.savez_compressed(a.out[:-4] + ".mediapipe.npz", **{k: (lm2 if k == "lm" else src_npz[k]) for k in src_npz.files}); rep.update(redrawn=done, lowconf=lowconf, seconds=round(time.perf_counter() - t0, 1))
    pathlib.Path(a.out[:-4] + ".lip_snap.json").write_text(json.dumps(rep, indent=1))
    print(f"== lip_snap {a.out}: {done} re-drawn ({lowconf} low-confidence), {rep['seconds']} s")


if __name__ == "__main__":
    main()
