"""Geometric jaw closing of a SOURCE frame (2026-10-14 prototype).

Reviewer: where the dub is silent the mouth must not move ("дубляж молчит, а рот шевелится — такого быть не должно"); every earlier way of
showing a closed mouth over a talking source failed because a closed jaw from ANOTHER image (LatentSync's generation with a closed reference,
a frozen keyframe) was composited over the open source jaw (smeared contour, ghosts). Here the source frame itself is deformed: the lower lip,
the chin and the lower jaw contour are moved up by the measured mouth opening (piecewise-affine warp on the insightface 2d106 landmarks, the jaw
contour weighted by its height between the nose base and the chin, as a jaw rotates), everything above the mouth and around the face stays.
The pixels are the frame's own (same view, same light, same background), so there is no second contour. LatentSync with the silent audio can
then render closed lips on a closed jaw (no conflict)."""
from __future__ import annotations

import numpy as np

INNER_UP, INNER_LO = [62, 66, 70], [60, 54, 57]
BROW_L, BROW_R = [43, 48, 49, 51, 50], list(range(101, 106))
NOSE_BASE = [76, 77, 78, 79, 80, 82, 83, 84, 85]


def opening(lm: np.ndarray) -> tuple[float, float]:
    """(inner-lip gap in px, inter-brow unit in px)."""
    unit = float(np.linalg.norm(lm[BROW_R].mean(0) - lm[BROW_L].mean(0)))
    return float(np.mean(lm[INNER_LO, 1] - lm[INNER_UP, 1])), max(unit, 1e-6)


def close_jaw(frame: np.ndarray, lm: np.ndarray, closed_gap: float = 0.02, strength: float = 1.0) -> tuple[np.ndarray, dict]:
    """Returns (frame with the jaw closed, info). closed_gap: target inner-lip gap / unit (lips touching)."""
    from skimage.transform import PiecewiseAffineTransform, warp
    lm = np.asarray(lm, np.float64); gap, unit = opening(lm)
    drop = max(0.0, gap - closed_gap * unit) * strength
    info = {"gap_px": round(gap, 2), "unit_px": round(unit, 2), "drop_px": round(drop, 2)}
    if drop < 0.5:
        return frame, info
    H, W = frame.shape[:2]
    c = lm[:33]; nose_y = float(lm[NOSE_BASE, 1].mean()); chin_y = float(c[:, 1].max()); x1, x2 = float(c[:, 0].min()), float(c[:, 0].max())
    fw, fh = x2 - x1, chin_y - float(lm[BROW_L + BROW_R, 1].mean())
    mouth = lm[52:72]; mid_y = 0.5 * (lm[INNER_UP, 1].mean() + lm[INNER_LO, 1].mean())
    src, dy = [], []
    for k in range(106):
        x, y = lm[k]
        if 52 <= k < 72:
            if k in (52, 61):                      # mouth corners: half way
                f = 0.5
            else:
                f = 1.0 if y > mid_y else 0.0
        elif k < 33:
            f = float(np.clip((y - nose_y) / max(chin_y - nose_y, 1e-6), 0.0, 1.0)) ** 1.2 if y > nose_y else 0.0
        else:
            f = 0.0
        src.append((x, y)); dy.append(-drop * f)
    # the neck under the chin follows half way (skin stretch), a fixed ring around the face and the ROI corners hold the rest
    for t in np.linspace(-0.4, 0.4, 5):
        src.append((0.5 * (x1 + x2) + t * fw, chin_y + 0.18 * fh)); dy.append(-0.5 * drop * (1 - abs(t)))
    ya, yb = max(0, int(nose_y - 0.25 * fh)), min(H, int(chin_y + 0.6 * fh)); xa, xb = max(0, int(x1 - 0.35 * fw)), min(W, int(x2 + 0.35 * fw))
    for x in np.linspace(xa, xb - 1, 7):
        for y in (ya, yb - 1):
            src.append((x, y)); dy.append(0.0)
    for y in np.linspace(ya, yb - 1, 6):
        for x in (xa, xb - 1):
            src.append((x, y)); dy.append(0.0)
    src = np.array(src); dst = src.copy(); dst[:, 1] += np.array(dy)
    roi = frame[ya:yb, xa:xb].astype(np.float32) / 255.0
    off = np.array([xa, ya]); t = PiecewiseAffineTransform()
    if not t.estimate(dst - off, src - off):          # output (closed) coords -> input (source) coords
        return frame, {**info, "error": "estimate failed"}
    out = warp(roi, t, output_shape=roi.shape, mode="edge", preserve_range=True)
    res = frame.copy(); res[ya:yb, xa:xb] = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return res, {**info, "roi": [xa, ya, xb, yb]}
