"""--silent-mouth freeze (2026-10-14): what the face does while the SOURCE speaker talks but the DUB is silent.

Human feedback on the thought pause policy with LatentSync + a closed-mouth reference in those frames: "оригинал говорит, дубляж молчит —
надо бороться с размазыванием ебальников, это треш". Cause (04 R1, 22-35 s): LatentSync draws a closed jaw while the source jaw is open; the
mouth-only paste-back ellipse is wider than the face, so inside it the cheek / jaw contour comes from the generated face and outside it from
the source -> a step / smear in the jaw contour where the two disagree. In the 2026-10-12 pipeline the generated jaw followed the source jaw
(reference = the frame itself), so the contours agreed, but the mouth moved without sound.

Freeze: no LatentSync in those frames. The lower face INCLUDING the jaw contour comes from ONE nearby source frame whose mouth is closed
(keyframe), warped onto the current frame with a similarity transform fitted on the stable points (eyes, brows, nose), composited with a
feathered mask = union of the keyframe's and the current lower-face contours (below the nose, dilated). One image provides the whole contour
-> no step; the mouth is closed and still; head, eyes and brows stay the current source frame. Runs longer than `chunk_s` get a new keyframe
per chunk with a crossfade; runs fade in / out over `ramp` frames where they meet plain source frames (LatentSync boundaries are blended by
the assembly crossfade, which now blends with the frozen frame)."""
from __future__ import annotations

import numpy as np

FPS = 25
CONTOUR = list(range(0, 33))                                                      # insightface 2d106 face contour (checked on 04, 2026-10-14)
STABLE = list(range(33, 52)) + list(range(72, 87)) + list(range(87, 106))         # eyes, brows, nose
NOSE_BASE = [76, 77, 78, 79, 80, 82, 83, 84, 85]
INNER = [(62, 60), (66, 54), (70, 57)]


def _open(lm: np.ndarray) -> float:
    unit = np.linalg.norm(lm[97:106].mean(0) - lm[43:52].mean(0))
    return float(np.mean([lm[b, 1] - lm[a, 1] for a, b in INNER]) / max(unit, 1e-6))


def _mask_of(intervals, n: int, pad: float) -> np.ndarray:
    m = np.zeros(n, bool)
    for a, b in intervals:
        m[max(0, int(np.floor((a - pad) * FPS))):min(n, int(np.ceil((b + pad) * FPS)))] = True
    return m


def _lip_thickness(lm: np.ndarray) -> float:
    """Outer lip height / inter-brow distance: pressed lips ("m / p / b", rolled in) are thin -> a strained face as a frozen mouth (01 R6)."""
    unit = np.linalg.norm(lm[97:106].mean(0) - lm[43:52].mean(0))
    return float((lm[52:72, 1].max() - lm[52:72, 1].min()) / max(unit, 1e-6))


def _droop(lm: np.ndarray) -> float:
    """Mouth corners below the lip centre line / inter-brow distance (> 0 = corners turned down: pressed / frowning mouth, the "strained" face of
    01 R6/R8 3.9-5.0 s, keyframe taken mid-speech on a pressed "m")."""
    unit = np.linalg.norm(lm[97:106].mean(0) - lm[43:52].mean(0))
    return float((lm[[52, 61], 1].mean() - 0.5 * (lm[71, 1] + lm[53, 1])) / max(unit, 1e-6))


def shots(frames: list, thr: float = 0.55) -> np.ndarray:
    """Shot id per frame: a cut where the HSV histogram correlation of consecutive frames drops below thr (01 cuts from the speaker to the
    interior b-roll and back to another room: a keyframe must come from the SAME shot)."""
    import cv2
    ids = np.zeros(len(frames), np.int32); prev = None; k = 0
    for i, f in enumerate(frames):
        h = cv2.calcHist([cv2.cvtColor(cv2.resize(f, (96, 96)), cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 16], [0, 180, 0, 256]); cv2.normalize(h, h)
        if prev is not None and cv2.compareHist(prev, h, cv2.HISTCMP_CORREL) < thr: k += 1
        ids[i] = k; prev = h
    return ids


def _upper_mask(shape, lm: np.ndarray) -> np.ndarray:
    """uint8 mask of the rigid upper head (brows, eyes / glasses, nose, upper cheeks) for the ECC alignment: no mouth / jaw."""
    import cv2
    pts = np.concatenate([lm[STABLE], lm[CONTOUR][lm[CONTOUR][:, 1] < lm[NOSE_BASE, 1].mean()]])
    m = np.zeros(shape[:2], np.uint8); cv2.fillConvexPoly(m, cv2.convexHull(pts.astype(np.int32)), 255)
    return m


def _sharpness(gray: np.ndarray, m: np.ndarray) -> float:
    import cv2
    v = cv2.Laplacian(gray, cv2.CV_32F)[m > 0]
    return float(v.var()) if v.size else 0.0


def _cavity(frame, lm: np.ndarray) -> float:
    """Dark mouth cavity / teeth line between the lips: mean grey of the inner-lip polygon relative to the lip ring (1.0 = no gap; lower =
    a visible gap). The landmark gap alone missed slightly parted lips of 04 (R6 2026-10-14)."""
    import cv2
    inner = lm[[62, 66, 54, 60, 57, 70]].astype(np.int32); outer = lm[52:72].astype(np.int32)
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); m_in = np.zeros(g.shape, np.uint8); m_out = np.zeros(g.shape, np.uint8)
    cv2.fillConvexPoly(m_in, cv2.convexHull(inner), 1); cv2.fillConvexPoly(m_out, cv2.convexHull(outer), 1); ring = (m_out > 0) & (m_in == 0)
    if m_in.sum() < 3 or ring.sum() < 3:
        return 1.0
    return float(np.percentile(g[m_in > 0], 20) / max(1.0, np.median(g[ring])))


def _lower_face_mask(shape, contours: list, nose_y: float, face_w: float) -> np.ndarray:
    import cv2
    H, W = shape[:2]; pts = []
    for c in contours:
        below = c[c[:, 1] >= nose_y]
        pts.append(below)
        if len(c):   # close the region at the nose base across the full contour width
            pts.append(np.array([[c[:, 0].min(), nose_y], [c[:, 0].max(), nose_y]]))
    hull = cv2.convexHull(np.concatenate(pts).astype(np.float32)).astype(np.int32)
    m = np.zeros((H, W), np.float32); cv2.fillConvexPoly(m, hull, 1.0)
    d = max(3, int(0.04 * face_w)); m = cv2.dilate(m, np.ones((d, d), np.uint8))
    k = max(5, int(0.16 * face_w) | 1)   # wide feather: no visible seam where keyframe and current skin meet
    return cv2.GaussianBlur(m, (k, k), 0)[..., None]


def apply_freeze(master_frames: list, ls_frames: np.ndarray, src_speech: list, dub_speech: list, detector, pad: float = 0.12, src_pad: float = 0.1,
                 window_s: float = 3.0, chunk_s: float = 3.0, ramp: int = 3, log=print) -> dict:
    """In place on master_frames (BGR): frames outside LatentSync (ls_frames False) where the source speaks and the dub is silent get the frozen
    closed-mouth lower face. Returns stats."""
    import cv2
    n = len(master_frames)
    want = _mask_of(src_speech, n, src_pad) & ~_mask_of(dub_speech, n, pad) & ~ls_frames
    st = {"frames_wanted": int(want.sum()), "frames_frozen": 0, "runs": 0, "keyframes": [], "no_face": 0, "no_key": 0}
    if not want.any():
        return st
    lms: dict = {}

    def lm(i):
        if i not in lms:
            bb, l = detector(cv2.cvtColor(master_frames[i], cv2.COLOR_BGR2RGB))
            lms[i] = None if bb is None else np.asarray(l, np.float32)
        return lms[i]

    # closed level of this video: 10th percentile of the opening over ~200 sampled frames with a face (R6 on 04 with "20th percentile of the
    # source-silent frames + 0.01" picked keyframes with slightly parted lips: the speaker smiles in his pauses)
    samp = [lm(int(i)) for i in range(0, n, max(1, n // 200))]
    ops = [_open(l) for l in samp if l is not None]
    thr = float(np.percentile(ops, 10)) if ops else None
    thick = [_lip_thickness(l) for l in samp if l is not None]; thick_med = float(np.median(thick)) if thick else None
    runs = []; i = 0
    while i < n:
        if want[i]:
            j = i
            while j < n and want[j]: j += 1
            runs.append((i, j)); i = j
        else:
            i += 1
    st["runs"] = len(runs); st["closed_thr"] = round(thr, 3) if thr is not None else None
    src_quiet = ~_mask_of(src_speech, n, 0.15)   # the speaker is silent there (+150 ms): his mouth is at rest, the most reliable "closed" frames
    src_quiet_tight = ~_mask_of(src_speech, n, 0.04)   # short breaths between sentences (01 has one 0.2 s pause in its first shot)
    shot = shots(master_frames); droops = [_droop(l) for l in samp if l is not None]; droop_med = float(np.median(droops)) if droops else 0.0
    for a, b in runs:
        chunks = []; c0 = a; L = int(chunk_s * FPS)
        while c0 < b:
            c1 = min(b, c0 + L)
            if b - c1 < L // 3: c1 = b
            chunks.append((c0, c1)); c0 = c1
        keys = []
        for c0, c1 in chunks:
            mid = (c0 + c1) // 2; ref = lm(mid) if lm(mid) is not None else next((lm(t) for t in range(c0, c1) if lm(t) is not None), None)
            if ref is None:
                keys.append(None); continue
            best = None; same = shot == shot[mid]
            # pass 1: only frames where the speaker is silent (mouth at rest), widening up to the whole shot; pass 2: any frame of the shot
            for quiet, w_ in [(src_quiet, window_s), (src_quiet, 2 * window_s), (src_quiet_tight, 2 * window_s), (None, window_s), (None, 2 * window_s), (None, n / FPS)]:
              lo, hi = max(0, c0 - int(w_ * FPS)), min(n, c1 + int(w_ * FPS))
              for t in range(lo, hi, 1 if quiet is not None else 2):
                if not same[t] or (quiet is not None and not quiet[t]): continue
                lt = lm(t)
                if lt is None: continue
                o = _open(lt)
                if thr is not None and o > thr: continue
                M, _ = cv2.estimateAffinePartial2D(lt[STABLE], ref[STABLE])
                if M is None: continue
                res = float(np.linalg.norm((lt[STABLE] @ M[:, :2].T + M[:, 2]) - ref[STABLE], axis=1).mean())
                fw = float(ref[CONTOUR, 0].max() - ref[CONTOUR, 0].min())
                if res > 0.05 * fw and w_ < n / FPS: continue          # another head pose (out-of-plane): the warp cannot match it (04 25.4 s ghost of the nose shadow)
                # closedness first (landmark gap, then a dark gap between the lips), then pose match (relative to face width), nearness, source silent
                # pressed lips are THINNER than usual (01: every "closed" frame of a speaker without pauses is a pressed "m" -> strained face);
                # a slight gap of relaxed lips is natural, so the dark-gap term is weak
                pressed = max(0.0, 0.95 - _lip_thickness(lt) / thick_med) if thick_med else 0.0
                droop = max(0.0, _droop(lt) - droop_med)                                             # corners lower than usual = frowning / strained
                score = 100 * o + 1.5 * (1.0 - min(1.0, _cavity(master_frames[t], lt))) + 40 * res / fw + 0.3 * abs(t - mid) / FPS + (0.0 if src_quiet[t] else 3.0) + 60 * pressed + 60 * droop
                if best is None or score < best[0]: best = (score, t)
              if best is not None: break
            keys.append(best[1] if best else None)
        st["keyframes"].append([{"chunk": [c0 / FPS, c1 / FPS], "key": (k / FPS if k is not None else None)} for (c0, c1), k in zip(chunks, keys)])
        kidx = {}   # frame -> its chunk's keyframe
        for (c0, c1), k in zip(chunks, keys):
            for t in range(c0, c1): kidx[t] = k
        def composite(t, k):
            if k is None or lm(t) is None:
                return None
            kf, kl, lt = master_frames[k], lm(k), lm(t)
            M, _ = cv2.estimateAffinePartial2D(kl[STABLE], lt[STABLE])
            if M is None:
                return None
            H, W = kf.shape[:2]
            # landmark similarity -> ECC refinement on the rigid upper head (R6 04 8-12 s: landmark noise behind sunglasses shook the frozen jaw)
            cg = cv2.cvtColor(master_frames[t], cv2.COLOR_BGR2GRAY).astype(np.float32); kg = cv2.cvtColor(kf, cv2.COLOR_BGR2GRAY).astype(np.float32)
            Winv = cv2.invertAffineTransform(M).astype(np.float32); um = _upper_mask(cg.shape, lt)
            try:
                _, Winv = cv2.findTransformECC(cg, kg, Winv, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), um, 5)
                M = cv2.invertAffineTransform(Winv)
            except cv2.error:
                pass
            wk = cv2.warpAffine(kf, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            # motion blur of the handheld frame: blur the sharp keyframe until its upper-head sharpness matches the current frame's
            sc = _sharpness(cg, um); wg = cv2.cvtColor(wk, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if sc > 0 and _sharpness(wg, um) > 1.3 * sc:
                for sg in (0.6, 0.9, 1.3, 1.8, 2.5):
                    wb = cv2.GaussianBlur(wk, (0, 0), sg)
                    if _sharpness(cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY).astype(np.float32), um) <= 1.15 * sc: break
                wk = wb
            kc = kl[CONTOUR] @ M[:, :2].T + M[:, 2]; cc = lt[CONTOUR]
            top = 0.5 * (float(lt[NOSE_BASE, 1].mean()) + float(lt[[63, 71, 67], 1].mean()))   # mid upper lip: nose / nostril shading stay the current frame
            fw = float(cc[:, 0].max() - cc[:, 0].min())
            m = _lower_face_mask(kf.shape, [kc, cc], top, fw)
            wk = wk.astype(np.float32)
            # illumination of the CURRENT frame (handheld outdoor video: exposure / shade change within a second; R6 04 36.8 s showed a brightness
            # step across the face): low-frequency difference current - keyframe, estimated outside both mouths, added to the keyframe
            cur = master_frames[t].astype(np.float32); wgt = (m[..., 0] > 0.05).astype(np.float32)
            for L in (lt, kl @ M[:, :2].T + M[:, 2]):
                cv2.fillConvexPoly(wgt, cv2.convexHull(L[52:72].astype(np.int32)), 0.0)
            sig = max(3.0, 0.12 * fw); W_ = cv2.GaussianBlur(wgt, (0, 0), sig) + 1e-4
            lf = cv2.GaussianBlur((cur - wk) * wgt[..., None], (0, 0), sig) / W_[..., None]
            wk = wk + lf * (m > 0.01)
            return wk, m

        bounds = [c1 for (_, c1) in chunks[:-1]]
        for t in range(a, b):
            k = kidx.get(t)
            if k is None:
                st["no_key"] += 1; continue
            c = composite(t, k)
            if c is None:
                st["no_face"] += 1; continue
            wk, m = c
            for c1 in bounds:   # crossfade to the neighbouring chunk's keyframe around a chunk boundary
                if c1 - ramp <= t < c1 + ramp:
                    other = kidx.get(c1 if t < c1 else c1 - 1)
                    o = composite(t, other) if other != k else None
                    if o is not None:
                        w = (t - (c1 - ramp) + 1) / (2 * ramp + 1)   # weight of the LATER chunk
                        later, earlier = ((o, (wk, m)) if t < c1 else ((wk, m), o))
                        wk = later[0] * w + earlier[0] * (1 - w); m = np.maximum(later[1], earlier[1])
            alpha = 1.0   # fade in / out where the run meets plain source frames (LatentSync boundaries are blended by the assembly crossfade)
            if a > 0 and not ls_frames[a - 1]: alpha = min(alpha, (t - a + 1) / (ramp + 1))
            if b < n and not ls_frames[b]: alpha = min(alpha, (b - t) / (ramp + 1))
            m = m * alpha; cur = master_frames[t].astype(np.float32)
            master_frames[t] = (wk * m + cur * (1 - m)).round().clip(0, 255).astype(np.uint8); st["frames_frozen"] += 1
    log(f"  [freeze] {st['frames_frozen']}/{st['frames_wanted']} frames frozen in {st['runs']} runs (no face {st['no_face']}, no keyframe {st['no_key']}, closed thr {st.get('closed_thr')})")
    return st

