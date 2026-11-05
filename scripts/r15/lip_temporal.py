#!/usr/bin/env python3
"""r15 F3: temporal stabilisation of the generated MOUTH in the LatentSync layer (user: "lip blinking kills it"), CPU, generic, no training.
Per layer frame t, in a box around its mouth (its own landmarks), the neighbours t-R..t+R are warped onto frame t with DIS optical flow
(grey crops t -> n); a neighbour pixel counts only where the forward-backward flow error is small (occlusion / big motion / a cut: the frame
itself is kept there).
  --mode color   LIP COLOUR LOCK: on the lip ring (outer minus inner lip polygon, feathered), the LOW-frequency colour (L, a*, b*,
                 normalised Gaussian blur inside the ring, sigma = --sigma x mouth width) is replaced by its per-pixel median over the
                 aligned neighbours; the per-frame structure / HF (and the inner mouth, teeth) stay the frame's own. Lip redness /
                 brightness cannot legitimately jump between frames, the shape can.
  --mode median  FLOW-ALIGNED TEMPORAL MEDIAN: the whole mouth box (lips + inner mouth, dilated) is replaced by the per-pixel median of
                 the aligned frames (BGR), weighted by the share of valid neighbours: single-frame pops / shimmer go, the mouth shape
                 follows the frame (the flow aligns the neighbours to it), no low-pass blur.
  --mode both    colour lock, then the median.
    python scripts/r15/lip_temporal.py --ls LAYER.mp4 --lm LAYER.mediapipe.npz --out OUT.mp4 --mode color [--radius 3]
"""
import argparse, json, subprocess, time
from collections import deque
import numpy as np, cv2

OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
cv2.setNumThreads(2)


def box_of(p, W, H):
    mw = float(np.linalg.norm(p[61] - p[291])); c = p[OUTER].mean(0)
    x0, x1 = int(c[0] - 0.95 * mw), int(c[0] + 0.95 * mw); y0, y1 = int(c[1] - 0.75 * mw), int(c[1] + 0.75 * mw)
    return max(0, x0), max(0, y0), min(W, x1), min(H, y1), mw


def nblur(x, m, s):   # Gaussian blur of x inside the mask m only (normalised convolution: no bleeding from skin / teeth)
    k = (0, 0); num = cv2.GaussianBlur(x * m[..., None], k, s); den = cv2.GaussianBlur(m, k, s)[..., None]
    return num / np.maximum(den, 1e-3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ls", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["color", "median", "both", "none"], default="color"); ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--median-gate", type=float, default=0.0, help="r15 F3: apply the flow median only where the frame's mouth pop (vs the mean of its "
                    "aligned neighbours) is >= this x its running median (full at +0.5); 0 = the median everywhere")
    ap.add_argument("--outlier", type=float, default=0.0, help="r15 F3 (old03 'explosions'): after the mode, on the WHOLE mouth (lips + teeth / "
                    "cavity, dilated 0.15 MW) a pixel is replaced by the mean of its two flow-aligned neighbours (t-1, t+1) only where it differs "
                    "from BOTH by > k x MAD while they agree (< k x MAD); k = this value (e.g. 3); 0 = off")
    ap.add_argument("--mradius", type=int, default=2, help="the median's radius (mode median / both)")
    ap.add_argument("--sigma", type=float, default=0.10, help="low-frequency colour scale, x mouth width (~8 px at a 80 px mouth)")
    ap.add_argument("--fb-err", type=float, default=0.03, help="forward-backward flow error limit, x mouth width (min 1.5 px)")
    ap.add_argument("--clamp", type=float, nargs=3, default=[14.0, 7.0, 7.0], help="max |correction| of L, a*, b* (colour mode)")
    ap.add_argument("--crf", type=int, default=10); ap.add_argument("--preset", default="medium")
    a = ap.parse_args(); t0 = time.perf_counter()
    lm = np.load(a.lm)["lm"][..., :2].astype(np.float32); cap = cv2.VideoCapture(a.ls)
    W, H, fps = int(cap.get(3)), int(cap.get(4)), cap.get(cv2.CAP_PROP_FPS) or 25.0; N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
                            "-c:v", "libx264", "-crf", str(a.crf), "-preset", a.preset, "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    R = max(a.radius if a.mode in ("color", "both") else 0, a.mradius if a.mode in ("median", "both") else 0, 1 if a.outlier > 0 else 0)
    buf = deque(); nxt = 0; stats = {"frames": 0, "done": 0, "valid_share": [], "corr_L": [], "corr_a": [], "med_delta": [], "outlier_px": [], "gated": []}
    pops_ = deque(maxlen=75)   # the recent mouth pops (median gate reference, ~3 s)

    def read():
        nonlocal nxt
        ok, f = cap.read()
        if ok: buf.append((nxt, f)); nxt += 1
        return ok
    for _ in range(R + 1): read()
    hist = deque(maxlen=R)   # the ORIGINAL previous frames (the neighbours are always unprocessed frames)
    while buf:
        t, ft = buf[0]; out = ft.copy(); p = lm[t] if t < len(lm) else None
        if p is not None and np.isfinite(p[OUTER + INNER]).all():
            x0, y0, x1, y1, mw = box_of(p, W, H)
            if x1 - x0 > 16 and y1 - y0 > 16:
                ct = ft[y0:y1, x0:x1]; gt = cv2.cvtColor(ct, cv2.COLOR_BGR2GRAY); h, w = gt.shape
                gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)); e_thr = max(1.5, a.fb_err * mw)
                neigh = [(n, f) for n, f in list(hist) + list(buf)[1:]]
                warped = {}
                for n, fn in neigh:
                    cn = fn[y0:y1, x0:x1]; gn = cv2.cvtColor(cn, cv2.COLOR_BGR2GRAY)
                    f1 = dis.calc(gt, gn, None); f2 = dis.calc(gn, gt, None)
                    mx, my = gx + f1[..., 0], gy + f1[..., 1]
                    back = cv2.remap(f2, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    err = np.hypot(f1[..., 0] + back[..., 0], f1[..., 1] + back[..., 1])
                    wn = cv2.remap(cn, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    warped[n] = (wn, err < e_thr)
                q = p - [x0, y0]; ring = np.zeros((h, w), np.uint8); cv2.fillPoly(ring, [np.round(q[OUTER]).astype(np.int32)], 1)
                mouth = cv2.dilate(ring, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.10 * mw) + 1,) * 2)).astype(np.float32)
                cv2.fillPoly(ring, [np.round(q[INNER]).astype(np.int32)], 0); ring = ring.astype(np.float32)
                fe = max(1.0, 0.03 * mw); ringf = cv2.GaussianBlur(ring, (0, 0), fe); mouthf = cv2.GaussianBlur(mouth, (0, 0), fe)
                cur = ct.astype(np.float32)
                if a.mode in ("color", "both"):
                    sel = [warped[n] for n, _ in neigh if abs(n - t) <= a.radius]
                    lab_t = cv2.cvtColor(ct, cv2.COLOR_BGR2LAB).astype(np.float32); s = a.sigma * mw
                    lf_t = nblur(lab_t, ring, s); stack = [lf_t]
                    for wn, v in sel:
                        lf = nblur(cv2.cvtColor(wn, cv2.COLOR_BGR2LAB).astype(np.float32), ring * v, s); lf[~v] = np.nan; stack.append(lf)
                    med = np.nanmedian(np.stack(stack), 0); corr = np.nan_to_num(med - lf_t)
                    corr = np.clip(corr, -np.array(a.clamp, np.float32), np.array(a.clamp, np.float32)) * ringf[..., None]
                    lab2 = np.clip(lab_t + corr, 0, 255).astype(np.uint8); cur = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR).astype(np.float32)
                    if ring.sum() > 0: stats["corr_L"].append(float(np.abs(corr[..., 0])[ring > 0].mean())); stats["corr_a"].append(float(np.abs(corr[..., 1])[ring > 0].mean()))
                if a.mode in ("median", "both"):
                    sel = [warped[n] for n, _ in neigh if abs(n - t) <= a.mradius]; stack = [cur]; vc = np.ones((h, w), np.float32)
                    for wn, v in sel:
                        x = wn.astype(np.float32); x[~v] = np.nan; stack.append(x); vc += v
                    med = np.nanmedian(np.stack(stack), 0); share = (vc - 1) / max(1, len(sel))
                    wgt = (mouthf * np.clip((share - 0.25) / 0.5, 0, 1))[..., None]
                    if a.median_gate > 0 and (t - 1) in warped and (t + 1) in warped:   # r15 F3: the median only on POP frames (sync: a global median costs ~0.4 SyncNet)
                        ga = 0.5 * (warped[t - 1][0].astype(np.float32) + warped[t + 1][0].astype(np.float32))
                        pop_ = float(np.abs(cur - ga).mean(-1)[mouth > 0].mean()) if mouth.any() else 0.0
                        ref_ = float(np.median(pops_)) if len(pops_) >= 10 else pop_; pops_.append(pop_)
                        gate_ = float(np.clip((pop_ / max(ref_, 1e-3) - a.median_gate) / 0.5, 0, 1)); wgt = wgt * gate_; stats["gated"].append(gate_)
                    stats["med_delta"].append(float((np.abs(med - cur) * wgt).sum() / max(1.0, wgt.sum())))
                    cur = cur + wgt * (med - cur); stats["valid_share"].append(float(share[mouth > 0].mean()) if mouth.any() else 0.0)
                if a.outlier > 0 and (t - 1) in warped and (t + 1) in warped:   # r15 F3: single-frame pops of the whole mouth (teeth / cavity too)
                    (w1, v1), (w2, v2) = warped[t - 1], warped[t + 1]; w1f, w2f = w1.astype(np.float32), w2.astype(np.float32)
                    g = lambda x: x.mean(-1)
                    d1, d2, d12 = np.abs(g(cur) - g(w1f)), np.abs(g(cur) - g(w2f)), np.abs(g(w1f) - g(w2f))
                    mz = cv2.dilate(mouth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.05 * mw) + 1,) * 2)) > 0
                    s_ = 1.4826 * float(np.median(d12[mz])) + 1.0 if mz.any() else 99.0
                    o_ = (np.minimum(d1, d2) > a.outlier * s_) & (d12 < a.outlier * s_) & v1 & v2 & mz
                    om = cv2.GaussianBlur(cv2.dilate(o_.astype(np.float32), np.ones((3, 3), np.uint8)), (0, 0), 1.0)[..., None]
                    cur = cur + om * (0.5 * (w1f + w2f) - cur); stats["outlier_px"].append(float(o_.sum()) / max(1, mz.sum()))
                out[y0:y1, x0:x1] = np.clip(cur + 0.5, 0, 255).astype(np.uint8); stats["done"] += 1
        enc.stdin.write(out.tobytes()); stats["frames"] += 1
        hist.append(buf.popleft()); read()
    enc.stdin.close(); enc.wait(); cap.release()
    rep = {"frames": stats["frames"], "processed": stats["done"], "mode": a.mode, "seconds": round(time.perf_counter() - t0, 1),
           "corr_L_mean": round(float(np.mean(stats["corr_L"])), 2) if stats["corr_L"] else None, "corr_a_mean": round(float(np.mean(stats["corr_a"])), 2) if stats["corr_a"] else None,
           "median_delta_mean": round(float(np.mean(stats["med_delta"])), 2) if stats["med_delta"] else None,
           "valid_share_mean": round(float(np.mean(stats["valid_share"])), 2) if stats["valid_share"] else None,
           "outlier_px_share_mean": round(float(np.mean(stats["outlier_px"])), 4) if stats["outlier_px"] else None,
           "outlier_frames_gt1pct": int(sum(x > 0.01 for x in stats["outlier_px"])),
           "median_gated_frames_gt05": int(sum(x > 0.5 for x in stats["gated"])) if stats["gated"] else None, "args": vars(a)}
    open(a.out.rsplit(".", 1)[0] + ".lt.json", "w").write(json.dumps(rep, indent=1))
    print("== lip_temporal", json.dumps({k: v for k, v in rep.items() if k != "args"}), flush=True)


if __name__ == "__main__":
    main()
