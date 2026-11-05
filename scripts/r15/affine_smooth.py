#!/usr/bin/env python3
"""r15 Z: LS_AFFINE_SMOOTH=<w>[:<order>] (default off) — INPUT-CROP STABILISATION at the root. LatentSync's crop affine comes from 3 points
(the two eyebrow centres + the nose centre of the replayed 106 landmarks); the lower face is GENERATED in that crop and pasted back with
the same affine, so any landmark noise (and any eyebrow raise) moves the generated mouth relative to the real head = slide / jitter.
Here the segment's per-frame affine matrices are smoothed over time with a zero-lag Savitzky-Golay filter (window w frames, polynomial
order 2 by default: a centred least-squares fit, so a real head move (constant velocity or acceleration) passes with no lag while
frame-to-frame noise is removed), the input crops are re-warped with the smoothed matrices exactly as LatentSync's aligner does
(kornia warp_affine, bilinear, fill; then the LANCZOS resize to the model resolution), and the SAME smoothed matrices are used for the
paste-back (restore_video). Segments shorter than the window get a shorter odd window."""
import numpy as np
import torch


def _sg(x, w, order):
    from scipy.signal import savgol_filter
    n = len(x); w = min(w, n if n % 2 else n - 1)
    if w <= order + 1: return x
    return savgol_filter(x, w, order, axis=0, mode="interp")


def smooth(ip, video_frames, faces, affine, spec):
    import cv2, kornia
    w, order = (int(v) for v in (str(spec).split(":") + ["2"])[:2]); w += (w % 2 == 0)
    A = np.stack([(m.detach().float().cpu().numpy() if torch.is_tensor(m) else np.asarray(m, np.float32)).reshape(2, 3) for m in affine]).astype(np.float64)
    As = _sg(A.reshape(len(A), 6), w, order).reshape(-1, 2, 3)
    r = ip.restorer; dev, dt = r.device, r.dtype; fs = r.face_size; res = faces.shape[-1]; out = []
    for i, fr in enumerate(video_frames[:len(As)]):
        img = torch.from_numpy(np.ascontiguousarray(fr)).to(device=dev, dtype=dt).permute(2, 0, 1).unsqueeze(0)
        M = torch.from_numpy(As[i]).to(device=dev, dtype=dt).unsqueeze(0)
        cf = kornia.geometry.transform.warp_affine(img, M, (fs[1], fs[0]), mode="bilinear", padding_mode="fill", fill_value=r.fill_value)
        cf = cf.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        out.append(torch.from_numpy(cv2.resize(cf, (res, res), interpolation=cv2.INTER_LANCZOS4)).permute(2, 0, 1))
    new_aff = [torch.from_numpy(As[i]).to(device=dev, dtype=dt).unsqueeze(0) for i in range(len(As))]
    d = np.abs(As - A); print(f"  [affine-smooth] SG w={w} order={order}: {len(As)} frames, mean |dA| translation {d[:, :, 2].mean():.2f} px, linear {d[:, :, :2].mean():.4f}", flush=True)
    return torch.stack(out).to(faces.dtype), new_aff
