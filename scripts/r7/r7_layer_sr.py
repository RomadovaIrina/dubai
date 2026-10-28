#!/usr/bin/env python3
"""r7 F (user: 'blurred / soapy lower face'): the LatentSync layer's mouth region is the diffusion VAE's output, ~0.73 of the source's sharpness
on the lips whatever the composite does (r7_sharp_probe). This restores detail ON THE LAYER before the composite: Real-ESRGAN x2 (BSD-3 code
and weights, RRDBNet re-implemented here in plain torch: no basicsr / facexlib / gfpgan dependency, no face detector) on a crop around the
layer's own lips, resized back to 1x (area) and blended inside a feathered lips-hull mask; the rest of the layer is untouched.
    python scripts/r7/r7_layer_sr.py --ls-raw X_ls_raw.mp4 --lm X_ls_raw.mediapipe.npz --out X_ls_sr.mp4 [--dilate 0.5] [--feather 0.15] [--strength 1.0]
Weights: models/realesrgan/RealESRGAN_x2plus.pth (sha256 49fafd45...266abb, 64 MB).
"""
from __future__ import annotations
import argparse, hashlib, pathlib, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "models/realesrgan/RealESRGAN_x2plus.pth"; SHA = "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


def rrdbnet(scale=2):
    import torch, torch.nn as nn, torch.nn.functional as F

    class RDB(nn.Module):
        def __init__(s, nf=64, gc=32):
            super().__init__(); s.c1 = nn.Conv2d(nf, gc, 3, 1, 1); s.c2 = nn.Conv2d(nf + gc, gc, 3, 1, 1); s.c3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
            s.c4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1); s.c5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1); s.l = nn.LeakyReLU(0.2, True)
        def forward(s, x):
            x1 = s.l(s.c1(x)); x2 = s.l(s.c2(torch.cat((x, x1), 1))); x3 = s.l(s.c3(torch.cat((x, x1, x2), 1)))
            x4 = s.l(s.c4(torch.cat((x, x1, x2, x3), 1))); x5 = s.c5(torch.cat((x, x1, x2, x3, x4), 1)); return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(s, nf, gc=32):
            super().__init__(); s.rdb1 = RDB(nf, gc); s.rdb2 = RDB(nf, gc); s.rdb3 = RDB(nf, gc)
        def forward(s, x): return s.rdb3(s.rdb2(s.rdb1(x))) * 0.2 + x

    class Net(nn.Module):
        def __init__(s, nf=64, nb=23, gc=32):
            super().__init__(); s.scale = scale; ic = 3 * (4 if scale == 2 else 1)
            s.conv_first = nn.Conv2d(ic, nf, 3, 1, 1); s.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)]); s.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
            s.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1); s.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1); s.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1); s.conv_last = nn.Conv2d(nf, 3, 3, 1, 1); s.l = nn.LeakyReLU(0.2, True)
        def forward(s, x):
            if s.scale == 2: x = F.pixel_unshuffle(x, 2)
            f = s.conv_first(x); f = s.conv_body(s.body(f)) + f
            f = s.l(s.conv_up1(F.interpolate(f, scale_factor=2, mode="nearest"))); f = s.l(s.conv_up2(F.interpolate(f, scale_factor=2, mode="nearest")))
            return s.conv_last(s.l(s.conv_hr(f)))
    return Net()


def load_model(device="cuda"):
    import torch
    if hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() != SHA: raise SystemExit(f"{WEIGHTS}: sha256 mismatch")
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)   # tensors only: a downloaded pickle never runs code here
    sd = sd.get("params_ema", sd.get("params", sd))
    # basicsr names: body.N.rdbK.convJ -> ours body.N.rdbK.cJ
    sd = {k.replace("conv1", "c1").replace("conv2", "c2").replace("conv3", "c3").replace("conv4", "c4").replace("conv5", "c5") if ".rdb" in k else k: v for k, v in sd.items()}
    net = rrdbnet(2); missing = net.load_state_dict(sd, strict=True); net.eval().to(device).half(); return net


def main():
    import cv2, torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ls-raw", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dilate", type=float, default=0.5, help="lips hull dilation in mouth widths = the restored region"); ap.add_argument("--feather", type=float, default=0.15)
    ap.add_argument("--strength", type=float, default=1.0); ap.add_argument("--crf", type=int, default=10); ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1, help="phase 2: ESRGAN crops per forward pass (crops padded to the batch's max size, reflect)")
    ap.add_argument("--encode-preset", default="medium"); ap.add_argument("--mask", choices=["dilate", "down2"], default="dilate", help="phase 2: down2 = the same dilate-by-r + Gaussian edge computed at half resolution and resized (max |diff| 0.07, mean 0.004; 1.3 ms instead of 17 ms per frame)")
    ap.add_argument("--temporal", type=int, default=0, help="r7 F: the ADDED detail (sr - raw) is averaged over +-N layer frames, flow-aligned (DIS on the raw crops) "
                    "to frame k, before it is added: per-frame ESRGAN texture flickers (03: twitch +4.7 at N=0); N=1 = 3-frame average")
    a = ap.parse_args(); lm = np.load(a.lm)["lm"]; net = load_model(); t0 = time.perf_counter()
    import av
    c = av.open(a.ls_raw); s = c.streams.video[0]; s.thread_type = "AUTO"; o = av.open(a.out, "w")   # phase 2: threaded decode (38 s -> 3 s on 1309 frames)
    st = o.add_stream("libx264", rate=s.average_rate); st.pix_fmt = "yuv420p"; st.options = {"crf": str(a.crf), "preset": a.encode_preset}; T = {"decode": 0.0, "sr": 0.0, "flow": 0.0, "encode": 0.0}
    frames = []; crops = []; boxes = []   # per frame: (x0, y0, x1, y1, mask) or None; the raw crop and its ESRGAN detail (sr - raw)
    pending = []   # (frame index, crop float32 BGR, mask, box) awaiting the ESRGAN pass
    def flush():
        if not pending: return
        t1 = time.perf_counter(); hs = [p_[1].shape[0] + p_[1].shape[0] % 2 for p_ in pending]; ws = [p_[1].shape[1] + p_[1].shape[1] % 2 for p_ in pending]; Hm, Wm = max(hs), max(ws)
        ts = []
        for _, crop, _, _ in pending:
            ch, cw = crop.shape[:2]; ts.append(np.pad(crop, ((0, Hm - ch), (0, Wm - cw), (0, 0)), mode="reflect")[..., ::-1].copy())
        t = torch.from_numpy(np.stack(ts)).permute(0, 3, 1, 2).half().cuda() / 255.0
        with torch.no_grad(): sr_b = net(t).clamp(0, 1)
        sr_b = (sr_b.permute(0, 2, 3, 1).float().cpu().numpy()[..., ::-1] * 255.0)
        for (k_, crop, m_, box), sr in zip(pending, sr_b):
            ch, cw = crop.shape[:2]; sr = cv2.resize(sr[: 2 * ch, : 2 * cw], (cw, ch), interpolation=cv2.INTER_AREA); boxes[k_] = box + (m_,); crops[k_] = (crop, sr - crop)
        pending.clear(); T["sr"] += time.perf_counter() - t1
    td = time.perf_counter()
    for k, fr in enumerate(c.decode(s)):
        f = fr.to_ndarray(format="bgr24"); H, W = f.shape[:2]
        if k == 0: st.width, st.height = W, H
        if a.max_frames and k >= a.max_frames: break
        frames.append(f); boxes.append(None); crops.append(None)
        p = lm[k] if k < len(lm) else None
        if p is None or not np.isfinite(p[LIPS_OUTER]).all(): continue
        q = p[LIPS_OUTER, :2]; mw = float(q[:, 0].max() - q[:, 0].min()); pad = (a.dilate + a.feather + 0.2) * mw
        x0, y0 = int(max(0, q[:, 0].min() - pad)), int(max(0, q[:, 1].min() - pad)); x1, y1 = int(min(W, q[:, 0].max() + pad)), int(min(H, q[:, 1].max() + pad))
        if x1 - x0 < 16 or y1 - y0 < 16: continue
        crop = f[y0:y1, x0:x1].astype(np.float32); ch, cw = crop.shape[:2]
        hull = cv2.convexHull((q - [x0, y0]).astype(np.float32))[:, 0]; m = np.zeros((ch, cw), np.float32); cv2.fillPoly(m, [np.round(hull).astype(np.int32)], 1.0)
        r = int(max(1, a.dilate * mw)); sig = max(0.5, a.feather * mw / 2.563)
        if a.mask == "down2":   # the same morphology at half resolution (the ellipse kernel is O(r^2) per pixel at full resolution)
            ms = cv2.resize(m, ((cw + 1) // 2, (ch + 1) // 2), interpolation=cv2.INTER_AREA); rr = max(1, int(round(r / 2)))
            ms = cv2.dilate(ms, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))); ms = cv2.GaussianBlur(ms, (0, 0), max(0.5, sig / 2))
            m = cv2.resize(ms, (cw, ch), interpolation=cv2.INTER_LINEAR)
        else:
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))); m = cv2.GaussianBlur(m, (0, 0), sig)
        pending.append((k, crop, m, (x0, y0, x1, y1)))
        if len(pending) >= a.batch: flush()
    flush(); T["decode"] = time.perf_counter() - td - T["sr"]
    c.close(); n = len(frames); done = 0
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM) if a.temporal > 0 else None
    def warp(img, F):   # img sampled at x + F
        h, w = F.shape[:2]; gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        return cv2.remap(img, gx + F[..., 0], gy + F[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    for k in range(n):
        f = frames[k]
        if boxes[k] is not None:
            x0, y0, x1, y1, m = boxes[k]; crop, det = crops[k]; tf = time.perf_counter()
            if dis is not None:   # average the detail over the neighbours, each warped onto frame k (flow on the raw crops, same box)
                acc = det.copy(); wsum = np.ones(det.shape[:2], np.float32); g0 = cv2.cvtColor(np.clip(crop, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                for j in range(k - a.temporal, k + a.temporal + 1):
                    if j == k or j < 0 or j >= n or boxes[j] is None: continue
                    cj = frames[j][y0:y1, x0:x1].astype(np.float32)   # frame j in frame k's box (the mouth moves little between neighbours)
                    # the detail of frame j, computed in ITS box: resample it into this box (overlap only)
                    xj0, yj0, xj1, yj1, _ = boxes[j]; dj = np.zeros_like(det); ox0, oy0, ox1, oy1 = max(x0, xj0), max(y0, yj0), min(x1, xj1), min(y1, yj1)
                    if ox1 <= ox0 or oy1 <= oy0: continue
                    dj[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0] = crops[j][1][oy0 - yj0:oy1 - yj0, ox0 - xj0:ox1 - xj0]
                    gj = cv2.cvtColor(np.clip(cj, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY); F = dis.calc(g0, gj, None)   # k -> j
                    dw = warp(dj, F); cw_ = warp(np.ones(det.shape[:2], np.float32), F) * np.exp(-np.abs(warp(cj, F) - crop).mean(-1) / 12.0)   # low weight where the warp disagrees
                    acc += dw * cw_[..., None]; wsum += cw_
                det = acc / wsum[..., None]
            f[y0:y1, x0:x1] = np.clip(crop + (a.strength * m)[..., None] * det, 0, 255).round().astype(np.uint8); done += 1; T["flow"] += time.perf_counter() - tf
        te = time.perf_counter(); nf = av.VideoFrame.from_ndarray(f, format="bgr24")
        for pk in st.encode(nf): o.mux(pk)
        T["encode"] += time.perf_counter() - te
    te = time.perf_counter()
    for pk in st.encode(): o.mux(pk)
    o.close(); T["encode"] += time.perf_counter() - te
    print(f"== layer_sr {a.out}: {done}/{n} frames restored (temporal +-{a.temporal}, batch {a.batch}), {time.perf_counter() - t0:.1f} s = " + " ".join(f"{k} {v:.1f}" for k, v in T.items()))


if __name__ == "__main__":
    main()
