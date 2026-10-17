#!/usr/bin/env python3
"""Temporal filters for the MediaPipe landmarks that drive LatentSync's crop / affine and the mouth-mask bbox (track A, round 2, 2026-10-17).

Defect under study: the generated lower face slides left / right relative to the untouched upper face, mostly on camera shake. The landmarks
fed to LatentSync are (clean0) a centred +-2 frame box average: on real shake it cannot follow (a 5-tap box at 25 fps has a ZERO at 5 Hz), so
the crop stays still while the head moves. Raw landmarks follow the head but carry MediaPipe's detector noise. The filters below separate the
two by measuring the REAL rigid head motion without any detector: dense optical flow (OpenCV DIS, no learned weights) on the rigid upper face
(brows / eyes / glasses / nose bridge), a RANSAC similarity per frame pair.

  rigid_motion(frames, lm)            per frame t: 2x3 similarity H_t mapping frame t-1 -> t (identity at t = 0 / no face)
  filter_landmarks(lm, H, spec)       (T, 478, 2) -> (T, 478, 2), same NaN pattern; spec:
      raw                             per-frame MediaPipe (no smoothing)
      box:K                           centred moving average +-K frames (clean0 = box:2), the previous behaviour
      flow:S                          flow-driven: landmarks expressed in head-stabilised coordinates (cumulative H), Gaussian-smoothed there
                                      (sigma S frames, only detector noise / slow drift is removed), mapped back -> follows the rigid motion
                                      frame by frame, no lag
      kalman:Q:R                      constant-velocity Kalman + RTS smoother per coordinate (offline, zero phase); Q accel noise, R meas noise (px^2)
      euro:MINCUT:BETA                One-Euro filter (causal, streaming), cutoff in Hz at 25 fps
      sub:N:S                         MediaPipe only every N-th frame, the frames in between from the flow track (stabilised-coordinate
                                      linear interpolation), then flow:S
All filters work inside contiguous face runs (a gap restarts them).
"""
from __future__ import annotations
import numpy as np

FPS = 25.0
# rigid upper face (MediaPipe 478 indices): brows, eye corners / lids area, glasses bridge, nose bridge, forehead edge, temples
UPPER = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 336, 296, 334, 293, 300, 285, 295, 282, 283, 276, 33, 133, 362, 263, 6, 168, 197, 195,
         9, 8, 151, 108, 337, 69, 299, 143, 372, 116, 345, 111, 340, 117, 346, 118, 347, 50, 280, 4, 5, 51, 281]
EYE_IN = [159, 145, 386, 374, 468, 473]  # pupils / lid centres (excluded radius: blinks)


def runs(ok: np.ndarray):
    out, i, n = [], 0, len(ok)
    while i < n:
        if ok[i]:
            j = i
            while j < n and ok[j]: j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def _region_mask(lm, H, W, pad):
    import cv2
    pts = lm[UPPER]; x0, y0 = np.floor(pts.min(0)) - pad; x1, y1 = np.ceil(pts.max(0)) + pad
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull((pts - [x0, y0]).astype(np.int32)), 1)
    iod = float(np.linalg.norm(lm[33] - lm[263]))
    for k in EYE_IN:   # blinks / gaze are not head motion
        cv2.circle(m, tuple(int(v) for v in (lm[k] - [x0, y0])), max(2, int(0.12 * iod)), 0, -1)
    return m.astype(bool), (x0, y0, x1, y1)


def rigid_motion(frames, lm: np.ndarray, step: int = 3, return_info: bool = False):
    """frames: sequence of BGR frames (len T); lm: (T, 478, 2) MediaPipe landmarks (region definition only, never the motion).
    Returns H (T, 2, 3): similarity mapping frame t-1 coordinates -> frame t (identity where t-1 or t has no face). DIS flow (MEDIUM preset)
    on a grey crop around the upper face of frame t-1; RANSAC partial affine (rotation + scale + translation) on flow samples inside the
    upper-face hull (eyes excluded)."""
    import cv2
    T = len(frames); Hs = np.tile(np.array([[1, 0, 0], [0, 1, 0]], np.float64), (T, 1, 1)); info = np.zeros((T, 2))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    Hh, Ww = frames[0].shape[:2]; prev_g = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for t in range(1, T):
        g = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
        if np.isfinite(lm[t - 1, 0, 0]) and np.isfinite(lm[t, 0, 0]):
            iod = float(np.linalg.norm(lm[t - 1, 33] - lm[t - 1, 263]))
            m, (x0, y0, x1, y1) = _region_mask(lm[t - 1], Hh, Ww, pad=int(0.6 * iod))
            # flow on a padded box (context), samples only inside the hull
            X0, Y0, X1, Y1 = max(0, x0 - 16), max(0, y0 - 16), min(Ww, x1 + 16), min(Hh, y1 + 16)
            F = dis.calc(np.ascontiguousarray(prev_g[Y0:Y1, X0:X1]), np.ascontiguousarray(g[Y0:Y1, X0:X1]), None)
            ys, xs = np.nonzero(m[::step, ::step]); ys = ys * step + (y0 - Y0); xs = xs * step + (x0 - X0)
            if len(xs) >= 12:
                p0 = np.stack([xs + X0, ys + Y0], 1).astype(np.float32); p1 = p0 + F[ys, xs]
                M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=0.5, maxIters=2000, confidence=0.995)
                if M is not None:
                    Hs[t] = M; info[t] = (len(xs), float(inl.mean()))
        prev_g = g
    return (Hs, info) if return_info else Hs


def _apply(H, p):
    return p @ H[:, :2].T + H[:, 2]


def _inv(H):   # (2, 3) -> (2, 3)
    Ai = np.linalg.inv(H[:, :2]); return np.concatenate([Ai, -(Ai @ H[:, 2])[:, None]], 1)


def _compose(H2, H1):   # x -> H2(H1(x))
    A = H2[:, :2] @ H1[:, :2]; b = H2[:, :2] @ H1[:, 2] + H2[:, 2]; return np.concatenate([A, b[:, None]], 1)


def cumulative(H: np.ndarray, s: int, e: int):
    """C[k] maps frame-s coordinates -> frame (s + k) coordinates, k = 0..e-s-1."""
    C = [np.array([[1, 0, 0], [0, 1, 0]], np.float64)]
    for t in range(s + 1, e): C.append(_compose(H[t], C[-1]))
    return np.stack(C)


def gauss1d(x: np.ndarray, sigma: float, axis: int = 0) -> np.ndarray:
    """Gaussian smoothing along axis with reflect padding (centred, zero phase)."""
    if sigma <= 0 or x.shape[axis] < 2: return x.copy()
    r = int(np.ceil(3 * sigma)); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    x = np.moveaxis(x, axis, 0); n = len(x)
    idx = np.clip(np.concatenate([np.arange(r, 0, -1), np.arange(n), np.arange(n - 2, n - 2 - r, -1)]), 0, n - 1)
    xp = x[idx]; out = np.zeros_like(x, dtype=np.float64)
    for j, w in enumerate(k): out += w * xp[j:j + n]
    return np.moveaxis(out, 0, axis)


def _box(x, k):
    n = len(x); out = np.empty_like(x, dtype=np.float64)
    for i in range(n): out[i] = x[max(0, i - k):min(n, i + k + 1)].mean(0)
    return out


def _kalman_rts(z: np.ndarray, q: float, r: float) -> np.ndarray:
    """Constant-velocity Kalman filter + Rauch-Tung-Striebel smoother, vectorised over all coordinates. z: (n, D)."""
    n, D = z.shape; F = np.array([[1, 1], [0, 1]]); Q = q * np.array([[0.25, 0.5], [0.5, 1.0]])
    xs = np.zeros((n, D, 2)); Ps = np.zeros((n, 2, 2)); xp_ = np.zeros((n, D, 2)); Pp_ = np.zeros((n, 2, 2))
    x = np.stack([z[0], np.zeros(D)], 1); P = np.diag([r, 10 * r])
    for t in range(n):
        if t > 0:
            x = x @ F.T; P = F @ P @ F.T + Q
        xp_[t], Pp_[t] = x, P
        S = P[0, 0] + r; K = P[:, 0] / S
        x = x + (z[t] - x[:, 0])[:, None] * K[None, :]; P = P - np.outer(K, P[0, :])
        xs[t], Ps[t] = x, P
    out = xs.copy()
    for t in range(n - 2, -1, -1):
        G = Ps[t] @ F.T @ np.linalg.inv(Pp_[t + 1])
        out[t] = xs[t] + (out[t + 1] - xp_[t + 1]) @ G.T
    return out[:, :, 0]


def _one_euro(z: np.ndarray, mincut: float, beta: float, dcut: float = 1.0) -> np.ndarray:
    def alpha(c): tau = 1.0 / (2 * np.pi * c); te = 1.0 / FPS; return 1.0 / (1.0 + tau / te)
    out = np.empty_like(z, dtype=np.float64); x = z[0].astype(np.float64); dx = np.zeros_like(x); out[0] = x
    for t in range(1, len(z)):
        d = (z[t] - x) * FPS; ad = alpha(dcut); dx = ad * d + (1 - ad) * dx
        c = mincut + beta * np.abs(dx); a = alpha(c); x = a * z[t] + (1 - a) * x; out[t] = x
    return out


def filter_landmarks(lm: np.ndarray, H: np.ndarray | None, spec: str) -> np.ndarray:
    """(T, P, 2) landmarks -> filtered copy (NaN pattern kept; filters restart at every gap). See module doc for `spec`."""
    kind, *args = spec.split(":"); out = lm.astype(np.float64).copy(); ok = np.isfinite(lm[:, 0, 0])
    if kind == "raw": return out
    for s, e in runs(ok):
        seg = lm[s:e].astype(np.float64); n, P, _ = seg.shape
        if kind == "box":
            out[s:e] = _box(seg, int(args[0]))
        elif kind == "kalman":
            q, r = float(args[0]), float(args[1]); out[s:e] = _kalman_rts(seg.reshape(n, -1), q, r).reshape(n, P, 2)
        elif kind == "euro":
            out[s:e] = _one_euro(seg.reshape(n, -1), float(args[0]), float(args[1])).reshape(n, P, 2)
        elif kind in ("flow", "sub"):
            if H is None: raise ValueError(f"{spec}: needs the rigid motion H (rigid_motion)")
            C = cumulative(H, s, e); Ci = np.stack([_inv(c) for c in C])
            stab = np.einsum("tij,tpj->tpi", Ci[:, :, :2], seg) + Ci[:, None, :, 2]      # detections in frame-s (head-stabilised) coordinates
            if kind == "sub":   # detector only on every N-th frame (plus the last), linear interpolation in stabilised coordinates
                N = int(args[0]); key = np.unique(np.r_[np.arange(0, n, N), n - 1]); tt = np.arange(n)
                st2 = np.empty_like(stab)
                for p in range(P):
                    for c in range(2): st2[:, p, c] = np.interp(tt, key, stab[key, p, c])
                stab = st2; sig = float(args[1]) if len(args) > 1 else 0.0
            else:
                sig = float(args[0])
            stab = gauss1d(stab, sig, 0)
            out[s:e] = np.einsum("tij,tpj->tpi", C[:, :, :2], stab) + C[:, None, :, 2]
        else:
            raise ValueError(f"unknown anchor filter {spec!r}")
    return out


def ls_anchors(lm106: np.ndarray, rounded: bool = True) -> np.ndarray:
    """(T, 106, 2) -> (T, 3, 2) the three affine anchors exactly as LatentSync's ImageProcessor.affine_transform builds them."""
    a = np.stack([lm106[:, [43, 48, 49, 51, 50]].mean(1), lm106[:, 101:106].mean(1), lm106[:, [74, 77, 83, 86]].mean(1)], 1)
    return np.round(a) if rounded else a


def anchor_stats(anc: np.ndarray, H: np.ndarray, sel: np.ndarray) -> dict:
    """Crop-anchor jitter vs the source head motion. anc (T, 3, 2) anchors as LatentSync receives them; H rigid motion (DIS); sel bool (T,)
    frames to score (t and t-1 both selected, both finite). resid = anchor velocity minus the head-motion velocity at the anchor
    (H_t applied to the previous anchor): detector noise + failure to follow; head = |head-motion velocity| for scale;
    acc = |second difference| (the old crop-jitter number); hp_gain = slope of anchor vs head velocity after removing the low
    frequencies (sigma 2 frames): 1 = follows the shake, 0 = ignores it; lag_frames = argmax cross-correlation of those."""
    T = len(anc); ok = sel & np.isfinite(anc[:, 0, 0]); okv = ok.copy(); okv[1:] &= ok[:-1]; okv[0] = False
    head = np.full((T, 3, 2), np.nan); v = np.full((T, 3, 2), np.nan)
    for t in np.where(okv)[0]:
        head[t] = _apply(H[t], anc[t - 1]) - anc[t - 1]; v[t] = anc[t] - anc[t - 1]
    r = np.linalg.norm(v - head, axis=2).mean(1)[okv]; hv = np.linalg.norm(head, axis=2).mean(1)[okv]
    c3 = np.zeros(T, bool); c3[1:-1] = ok[:-2] & ok[1:-1] & ok[2:]
    acc = np.linalg.norm(anc[2:] - 2 * anc[1:-1] + anc[:-2], axis=2).mean(1)[c3[1:-1]]
    # high-pass gain / lag on x (left-right, the reported direction) of the anchor mean
    hx = np.nan_to_num(head[:, :, 0].mean(1)); vx = np.nan_to_num(v[:, :, 0].mean(1)); m = okv.astype(float)
    hxh = hx - gauss1d(hx, 2.0); vxh = vx - gauss1d(vx, 2.0); hxh *= m; vxh *= m
    gain = float((hxh * vxh).sum() / max((hxh ** 2).sum(), 1e-9))
    xc = [float((np.roll(vxh, -L) * hxh).sum()) for L in range(-3, 4)]; lag = int(np.argmax(xc)) - 3
    def st(x): return {"mean": round(float(np.mean(x)), 4), "p95": round(float(np.percentile(x, 95)), 4)} if len(x) else None
    return {"n": int(okv.sum()), "resid_vs_head_px": st(r), "head_vel_px": st(hv), "acc_px": st(acc), "hp_gain_x": round(gain, 3), "lag_frames_x": lag}


def ls_bbox_float(lm106: np.ndarray) -> np.ndarray | None:
    """clean_face_stack.ls_bbox (LatentSync FaceDetector's formula) without the integer rounding: (x1, y1, x2, y2) float."""
    from clean_face_stack import LMK_ADAPT_ORIGIN_ORDER
    half = 0.5 * (lm106[74] + lm106[73]); sub = lm106[LMK_ADAPT_ORIGIN_ORDER]
    dist = sub[:, 1].max() - half[1]; upper = half[1] - dist
    x1, y1, x2, y2 = sub[:, 0].min(), upper, sub[:, 0].max(), sub[:, 1].max()
    if y2 - y1 <= 0 or x2 - x1 <= 0: return None
    w = x2 - x1; return np.array([x1 - 0.05 * w, y1, x2 + 0.05 * w, y2 + 0.1 * w])


def head_locked_boxes(lm106: np.ndarray, H: np.ndarray, sig_rigid: float = 8.0, sig_chin: float = 1.0) -> np.ndarray:
    """Mouth-mask boxes (T, 4) that move with the HEAD: per face run, the float LatentSync box is expressed in head-stabilised coordinates
    (inverse cumulative rigid motion), its centre-x, width and top are smoothed with sigma_rigid (they only change with yaw / scale, slowly),
    its bottom (chin: jaw opening) with sigma_chin, then mapped back with the frame's rigid motion (centre by the similarity, size by its scale).
    NaN rows where there is no face."""
    T = len(lm106); out = np.full((T, 4), np.nan); ok = np.isfinite(lm106[:, 0, 0])
    for s, e in runs(ok):
        bx = np.array([ls_bbox_float(lm106[t]) if ls_bbox_float(lm106[t]) is not None else (np.nan,) * 4 for t in range(s, e)], np.float64)
        C = cumulative(H, s, e); sc = np.sqrt(np.abs(np.linalg.det(C[:, :, :2])))
        cen = np.stack([(bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2], 1)
        Ci = np.stack([_inv(c) for c in C]); cs = np.einsum("tij,tj->ti", Ci[:, :, :2], cen) + Ci[:, :, 2]
        w = (bx[:, 2] - bx[:, 0]) / sc; top = cs[:, 1] - (bx[:, 3] - bx[:, 1]) / 2 / sc; bot = cs[:, 1] + (bx[:, 3] - bx[:, 1]) / 2 / sc
        cx = gauss1d(cs[:, 0], sig_rigid); w = gauss1d(w, sig_rigid); top = gauss1d(top, sig_rigid); bot = gauss1d(bot, sig_chin)
        cst = np.stack([cx, (top + bot) / 2], 1); cb = np.einsum("tij,tj->ti", C[:, :, :2], cst) + C[:, :, 2]
        hw, hh = w * sc / 2, (bot - top) * sc / 2
        out[s:e] = np.stack([cb[:, 0] - hw, cb[:, 1] - hh, cb[:, 0] + hw, cb[:, 1] + hh], 1)
    return out
