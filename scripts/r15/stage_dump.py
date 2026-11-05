#!/usr/bin/env python3
"""r15 Z: LS_DUMP_DIR=<dir> (default off) — per LatentSync segment (ls_stable.StableLatentSync._core), dump the generator's view of the face
for the per-stage jitter decomposition (scripts/r15/stage_jitter.py):
  seg_<start>_in.mp4   the INPUT aligned crops (512, 25 fps) = the source face after LatentSync's affine (crop-space source)
  seg_<start>_out.mp4  the GENERATED crops before the restore (after LatentSync's own paste of the upper crop)
  seg_<start>.npz      affine (n, 2, 3) frame -> crop, boxes (n, 4), start (master frame of the segment)"""
import os, pathlib
import numpy as np


def _u8(x):
    import torch
    a = x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, np.float32)
    if a.ndim == 4 and a.shape[1] in (1, 3): a = a.transpose(0, 2, 3, 1)   # (f, c, h, w) -> (f, h, w, c)
    if a.min() < -0.01: a = (a + 1) / 2 * 255                              # [-1, 1]
    elif a.max() <= 1.01: a = a * 255                                       # [0, 1]
    return np.clip(a, 0, 255).round().astype(np.uint8)


def dump(d, start, faces, synced, boxes, affine):
    import imageio.v2 as iio
    d = pathlib.Path(d); d.mkdir(parents=True, exist_ok=True); tag = d / f"seg_{int(start):05d}"
    for name, x in (("in", faces), ("out", synced)):
        if x is None: continue
        a = _u8(x); w = iio.get_writer(f"{tag}_{name}.mp4", fps=25, codec="libx264", macro_block_size=None, quality=None,
                                     ffmpeg_params=["-crf", "10", "-preset", "fast", "-pix_fmt", "yuv420p"])
        for f in a: w.append_data(f)
        w.close()
    A = np.stack([np.asarray(m.detach().cpu().numpy() if hasattr(m, "detach") else m, np.float64).reshape(-1)[:6].reshape(2, 3) for m in affine])
    np.savez(f"{tag}.npz", affine=A, boxes=np.asarray(boxes, np.float64), start=int(start))
