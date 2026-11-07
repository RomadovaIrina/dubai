#!/usr/bin/env python3
"""r14 F night P2 (LS_REF_SMART=1 with --ref-mode fixed, default off): a 16-frame UNet reference clip picked from the SOURCE (master) frames
by a generic per-frame score, instead of the current frames (which copy the source's closed mouth into the dub = the 'lazy mouth').
    score = z(mouth opening: MediaPipe inner-lip gap 13-14 / face height 10-152) + z(teeth: share of bright pixels inside the inner-lip box)
          + z(sharpness: Laplacian variance of the mouth box) - z(|yaw|: nose vs eye-centre offset / IOD) - z(|roll|: eye-line angle)
          - z(|face luma - median|)
Frames need a face, opening > OPEN_MIN, |yaw| / |roll| under limits; the top 16 are taken greedily at least n / 48 frames apart (spread in time),
returned in time order. Fewer than 16 usable frames -> None (the caller keeps its default reference)."""
import numpy as np, cv2

OPEN_MIN, YAW_MAX, ROLL_MAX = 0.03, 0.12, 8.0


def _z(v):
    v = np.asarray(v, np.float64); s = np.nanstd(v)
    return (v - np.nanmedian(v)) / (s if s > 1e-9 else 1.0)


def mouth_box(f):
    """inner-lip box (x0, y0, x1, y1) of one MediaPipe 478-point frame, padded by 15 % of the mouth width."""
    xs, ys = f[[78, 308, 13, 14, 82, 312, 87, 317], 0], f[[78, 308, 13, 14, 82, 312, 87, 317], 1]
    w = xs.max() - xs.min(); p = 0.15 * w
    return int(xs.min() - p), int(ys.min() - p), int(xs.max() + p), int(ys.max() + p)


def frame_feats(img_bgr, f):
    """(teeth share, sharpness, face luma) of one frame from its landmarks; NaN without a face."""
    if f is None or not np.isfinite(f[0, 0]): return np.nan, np.nan, np.nan
    H, W = img_bgr.shape[:2]; x0, y0, x1, y1 = mouth_box(f); x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 8 or y1 - y0 < 4: return np.nan, np.nan, np.nan
    g = cv2.cvtColor(img_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    fx0, fy0 = int(max(0, f[:, 0].min())), int(max(0, f[:, 1].min())); fx1, fy1 = int(min(W, f[:, 0].max())), int(min(H, f[:, 1].max()))
    face = cv2.cvtColor(img_bgr[fy0:fy1:4, fx0:fx1:4], cv2.COLOR_BGR2GRAY).astype(np.float32)
    thr = np.percentile(face, 90) if face.size else 200.0
    return float((g > thr).mean()), float(cv2.Laplacian(g, cv2.CV_32F).var()), float(face.mean()) if face.size else np.nan


def geometry(lm):
    """per frame (opening, |yaw|, |roll| deg) from MediaPipe landmarks (N, 478, >=2)."""
    fh = np.linalg.norm(lm[:, 10, :2] - lm[:, 152, :2], axis=1); op = np.linalg.norm(lm[:, 13, :2] - lm[:, 14, :2], axis=1) / np.maximum(fh, 1e-6)
    le, re = lm[:, 33, :2], lm[:, 263, :2]; iod = np.maximum(np.linalg.norm(re - le, axis=1), 1e-6)
    yaw = np.abs(lm[:, 1, 0] - (le[:, 0] + re[:, 0]) / 2) / iod
    roll = np.abs(np.degrees(np.arctan2(re[:, 1] - le[:, 1], re[:, 0] - le[:, 0])))
    return op, yaw, roll


def pick(frames_bgr, lm, n_ref=16, feats=None):
    """frames_bgr: indexable / iterable of BGR frames aligned with lm (N, 478, 3) (or pass precomputed feats (N, 3)).
    -> (sorted frame indices, info) or (None, info)."""
    N = len(lm); op, yaw, roll = geometry(lm)
    if feats is None: feats = np.array([frame_feats(fr, lm[i]) for i, fr in enumerate(frames_bgr)])
    teeth, sharp, luma = feats[:, 0], feats[:, 1], feats[:, 2]
    ok = np.isfinite(op) & np.isfinite(teeth) & (op > OPEN_MIN) & (yaw < YAW_MAX) & (roll < ROLL_MAX)
    info = {"usable": int(ok.sum()), "frames": int(N)}
    if ok.sum() < n_ref: info["fallback"] = "too few usable frames"; return None, info
    s = _z(np.where(ok, op, np.nan)) + _z(np.where(ok, teeth, np.nan)) + _z(np.where(ok, np.log1p(sharp), np.nan)) \
        - _z(np.where(ok, yaw, np.nan)) - _z(np.where(ok, roll, np.nan)) - _z(np.where(ok, np.abs(luma - np.nanmedian(luma[ok])), np.nan))
    s = np.where(ok, s, -np.inf); gap = max(1, N // 48); chosen = []
    for i in np.argsort(-s):
        if not np.isfinite(s[i]): break
        if all(abs(int(i) - j) >= gap for j in chosen): chosen.append(int(i))
        if len(chosen) == n_ref: break
    if len(chosen) < n_ref: info["fallback"] = f"only {len(chosen)} spread frames"; return None, info
    chosen.sort(); info.update({"idx": chosen, "score_mean": round(float(np.mean(s[chosen])), 3), "open_mean": round(float(op[chosen].mean()), 4),
                                "open_src_median": round(float(np.nanmedian(op)), 4)})
    return chosen, info


def pick_from_video(video, lm, n_ref=16):
    """streaming version for a master video on disk (frames are not kept): -> (indices, info, the chosen BGR frames)."""
    cap = cv2.VideoCapture(str(video)); feats = []; i = 0
    while i < len(lm):
        ok, fr = cap.read()
        if not ok: break
        feats.append(frame_feats(fr, lm[i])); i += 1
    cap.release(); feats = np.array(feats + [(np.nan,) * 3] * (len(lm) - len(feats)))
    idx, info = pick(None, lm, n_ref, feats)
    if idx is None: return None, info, None
    cap = cv2.VideoCapture(str(video)); out = {}; i = 0; want = set(idx)
    while i <= max(idx):
        ok, fr = cap.read()
        if not ok: break
        if i in want: out[i] = fr
        i += 1
    cap.release()
    return idx, info, [out[k] for k in idx]
