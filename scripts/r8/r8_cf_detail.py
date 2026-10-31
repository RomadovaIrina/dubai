#!/usr/bin/env python3
"""r8 F: CodeFormer DETAIL onto the clean LatentSync layer (user 29.10: m14 CodeFormer looks better, but its lip-sync / mismatch are worse).

Two layers of the SAME generator run (seeded LatentSync, same face stack, same audio): BASE = the clean LatentSync layer (m15raw ls_raw),
CF = the same frames after CodeFormer (m16raw ls_raw). Inside a feathered mouth region (outer-lip hull of BASE + --dilate IOD):
    D = CF - BASE;  detail = D - Gauss(D, --sigma x mouth width)    (high frequencies only: lip line, teeth, texture)
    out = BASE + --alpha * mask * detail_t + --beta * mask * (D - detail)
where detail_t is detail averaged over frames t-1, t, t+1 after DIS-flow alignment of the neighbours to t (--temporal 1): CodeFormer is per-frame.
The low frequencies (mouth shape, opening, motion = what SyncNet and the mismatch see) stay LatentSync's unless --beta > 0.
    python scripts/r8/r8_cf_detail.py --base BASE.mp4 --cf CF.mp4 --lm BASE.mediapipe.npz --out OUT.mp4 [--alpha 1 --beta 0 --sigma 0.12 --temporal 1]
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import numpy as np, cv2
cv2.setNumThreads(4)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from r8_mouth_flicker import LIPS_OUTER, LIPS_INNER


def main():
    import av
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True); ap.add_argument("--cf", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=1.0); ap.add_argument("--beta", type=float, default=0.0); ap.add_argument("--sigma", type=float, default=0.12)
    ap.add_argument("--dilate", type=float, default=0.25); ap.add_argument("--inner-alpha", type=float, default=1.0, help="r9: detail weight inside the inner-lip polygon (teeth / cavity; CodeFormer draws them sharper and brighter than the source: m03 x1.5)"); ap.add_argument("--temporal", type=int, default=1); ap.add_argument("--crf", type=int, default=10); ap.add_argument("--preset", default="medium", help="x264 preset of this intermediate layer (r8 speed: veryfast)"); ap.add_argument("--threads", type=int, default=1, help="r9 speed: frames processed in parallel (same output)")
    a = ap.parse_args(); lm = np.load(a.lm)["lm"]; t0 = time.perf_counter()
    cb, cc = av.open(a.base), av.open(a.cf); sb = cb.streams.video[0]; sb.thread_type = "AUTO"; cc.streams.video[0].thread_type = "AUTO"
    rate = sb.average_rate   # r9: read while the container is open (after cb.close() the stream reads freed memory: random rate -> libx264 "timebase is not set", m15 / m19 / m20)
    B = [f.to_ndarray(format="bgr24") for f in cb.decode(video=0)]; C = [f.to_ndarray(format="bgr24") for f in cc.decode(video=0)]; cb.close(); cc.close()
    n = min(len(B), len(C), len(lm)); H, W = B[0].shape[:2]
    boxes = [None] * n; det = [None] * n; outside = []
    # r9 speed L2: the per-frame work runs on --threads threads (OpenCV releases the GIL; every frame's result depends only on its inputs, so the
    # output is the same as one thread: one DIS instance per thread, frames encoded in order)
    import concurrent.futures as cf, threading
    tl = threading.local()
    def dis_():
        if not hasattr(tl, "dis"): tl.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        return tl.dis
    def first(k):   # per frame: mouth box, mask, HF detail
        p = lm[k]
        if not np.isfinite(p[LIPS_OUTER + [33, 263]]).all(): return None
        iod = float(np.linalg.norm(p[33, :2] - p[263, :2])); mw = float(np.linalg.norm(p[61, :2] - p[291, :2])); r = int(0.9 * iod)
        cx, cy = p[LIPS_OUTER, 0].mean(), p[LIPS_OUTER, 1].mean(); y0, y1, x0, x1 = max(0, int(cy - r)), min(H, int(cy + r)), max(0, int(cx - r)), min(W, int(cx + r))
        if y1 - y0 < 16 or x1 - x0 < 16: return None
        b = B[k][y0:y1, x0:x1].astype(np.float32); c = C[k][y0:y1, x0:x1].astype(np.float32); D = c - b
        m = np.zeros((y1 - y0, x1 - x0), np.uint8); hull = cv2.convexHull((p[LIPS_OUTER, :2] - [x0, y0]).astype(np.float32))[:, 0]
        cv2.fillPoly(m, [np.round(hull).astype(np.int32)], 1); rd = max(1, int(a.dilate * iod))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rd + 1,) * 2)).astype(np.float32); m = cv2.GaussianBlur(m, (0, 0), max(1.0, 0.08 * iod))
        if a.inner_alpha < 1.0 and np.isfinite(p[LIPS_INNER]).all():
            mi = np.zeros_like(m, np.uint8); cv2.fillPoly(mi, [np.round(p[LIPS_INNER, :2] - [x0, y0]).astype(np.int32)], 1)
            m = m * (1.0 - (1.0 - a.inner_alpha) * cv2.GaussianBlur(mi.astype(np.float32), (0, 0), max(1.0, 0.03 * iod)))
        sg = max(0.8, a.sigma * mw); lf = cv2.GaussianBlur(D, (0, 0), sg)
        ring = (m < 0.02)
        return (y0, y1, x0, x1, m), (D - lf, lf), (float(np.abs(D[ring]).mean()) if ring.any() else 0.0)
    def second(k):
        img = B[k]
        if boxes[k] is not None:
            y0, y1, x0, x1, m = boxes[k]; hf, lf = det[k]; acc = hf.copy(); wsum = 1.0
            if a.temporal > 0:
                gb = cv2.cvtColor(B[k][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                for j in range(k - a.temporal, k + a.temporal + 1):
                    if j == k or j < 0 or j >= n or det[j] is None: continue
                    # the neighbour's detail at the same image box, flow-aligned to frame k (flow on the BASE crops)
                    gj = cv2.cvtColor(B[j][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY); F = dis_().calc(gb, gj, None)
                    jy0, jy1, jx0, jx1, _ = boxes[j]; hfj = np.zeros_like(hf)
                    oy0, oy1, ox0, ox1 = max(y0, jy0), min(y1, jy1), max(x0, jx0), min(x1, jx1)
                    if oy1 - oy0 < 8 or ox1 - ox0 < 8: continue
                    hfj[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0] = det[j][0][oy0 - jy0:oy1 - jy0, ox0 - jx0:ox1 - jx0]
                    gy, gx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
                    acc += cv2.remap(hfj, gx + F[..., 0], gy + F[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); wsum += 1.0
            add = (a.alpha * acc / wsum + a.beta * lf) * m[..., None]
            img = img.copy(); img[y0:y1, x0:x1] = np.clip(img[y0:y1, x0:x1].astype(np.float32) + add, 0, 255).round().astype(np.uint8)
        return img
    o = av.open(a.out, "w"); st = o.add_stream("libx264", rate=rate); st.width, st.height = W, H; st.pix_fmt = "yuv420p"; st.options = {"crf": str(a.crf), "preset": a.preset}
    with cf.ThreadPoolExecutor(max_workers=max(1, a.threads)) as ex:
        for k, r_ in enumerate(ex.map(first, range(n))):
            if r_ is not None: boxes[k], det[k] = r_[0], r_[1]; outside.append(r_[2])
        CH = 64   # bounded: at most CH finished frames wait for the in-order encoder
        for c0 in range(0, n, CH):
            for img in ex.map(second, range(c0, min(n, c0 + CH))):
                for pk in st.encode(av.VideoFrame.from_ndarray(img, format="bgr24")): o.mux(pk)
    for pk in st.encode(): o.mux(pk)
    o.close()
    rep = {"frames": n, "mouth_frames": int(sum(b is not None for b in boxes)), "cf_minus_base_outside_mouth_mad": round(float(np.median(outside)), 3) if outside else None,
           "seconds": round(time.perf_counter() - t0, 1), "args": vars(a)}
    pathlib.Path(a.out).with_suffix(".cfd.json").write_text(json.dumps(rep, indent=1)); print("== cf_detail", json.dumps({k: v for k, v in rep.items() if k != "args"}), flush=True)


if __name__ == "__main__":
    main()
