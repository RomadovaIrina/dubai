#!/usr/bin/env python3
"""High-resolution, face-stabilised mouth / face temporal-jitter metric ("the mouth twitches", 2026-10-13 review).

Why a new metric: face_temporal.py measures 64x64 grey crops of a fixed box and insightface landmarks of the OUTPUT; it rates the cand5
mouth as calm as the source while a human sees it twitch. This script measures the candidate and the reference on IDENTICAL,
face-stabilised, full-resolution crops and separates motion from appearance:

  stabilisation   YuNet (OpenCV FaceDetectorYN, MIT code + weights) eyes + nose of the REFERENCE frames, Gaussian-smoothed over time
                  (sigma --lm-sigma frames, detector noise only), similarity transform -> canonical face (inter-ocular distance --iod px);
                  stage 2: the high-frequency head / camera shake left in the crops is measured with DIS flow on the reference's rigid
                  region (brows / eyes / nose bridge), high-passed (--hp-sigma) and removed from both videos with the same warp.
                  The SAME transform is applied to both videos, so head motion cancels and a mouth that moves relative to the head is
                  measured in the same coordinates for both. Mouth / jaw positions never drive the alignment.
  motion          OpenCV DIS optical flow (no learned weights) between consecutive canonical crops, per video, MINUS the similarity
                  motion of the rigid head region (brows / eyes / nose bridge, RANSAC fit to the same flow) -> motion relative to the head:
                    vel   = mean |F(t-1->t)|                       how much the region moves (articulation)
                    jerk  = mean |F(t->t+1) - F(t-1->t)|           change of motion = twitch / jump (smooth articulation keeps it low)
  appearance      resid = mean |I(t) - warp(I(t-1), F)| on forward-backward-consistent pixels: appearance change NOT explained by
                          motion (texture popping, teeth / lip-texture flicker, blending flicker)
                  acc   = mean |I(t+1) - 2 I(t) + I(t-1)|          full-resolution pixel acceleration (the old flicker, without 64x64)
                  d1    = mean |I(t) - I(t-1)|                     full-resolution pixel change
                  sharp = mean |Laplacian(I(t))|                   detail / sharpness (a restorer trade-off: calm but blurry is not a win)
                  shear = |spatial gradient| of the head-relative flow (mean, p95): a seam where pasted and source skin move differently
  per second      resid / d1 / shear ratios, frozen_mouth = (lower-face motion / mouth motion) of the video over the same of the reference
                  (> 1: the mouth is stiller than the face around it, e.g. a closed dubbed mouth while the source jaw / cheeks talk), and
                  max_d1_jump (the largest single-frame mouth change of that second / the median): ranks 1 = worst
  regions         mouth (lips / teeth, from the reference's mouth corners), lowerface (jaw / chin / cheeks band around the mouth,
                  where the mouth-only paste-back mask edge lives), upper (brows / eyes: a mouth-only composite must stay at 1.0)
  indices         twitch  = sum(jerk) / sum(vel): jerky vs smooth motion, independent of how much the speaker articulates
                  impulse = sum(acc) / sum(d1): a change that happens in ONE frame (a snap open / shut, a 1-frame lip distortion) has
                            acc ~ 2 x d1; a change spread over 3-4 frames has acc << d1
                  pops    = share of frames whose d1 / acc exceeds the REFERENCE's 98th percentile, / 0.02 (reference = 1.0 by construction)
                  every value is reported for the video, the reference and as the ratio video / reference

Region classes (with --manifest of a run): ls_core (LatentSync frames > --win from a switch), switch (+-win frames of an LS<->original
switch), orig (pass-through). LatentSync window boundaries (with --manifest): LatentSync 1.6 denoises independent --ls-window (16) frame
windows from each LS segment start; per-phase mouth d1 / resid / vel / jerk and the jump at every window boundary, video vs reference.

    python scripts/quality/mouth_jitter_hr.py --video OUT.mp4 --reference work_03/master_25fps.mp4 --out x.json \
        [--frames 1000:1300 1500:1600] [--manifest RUN/final/03_final.manifest.json] [--tag cand5] [--series] [--sheet DIR]
CPU only, deterministic. Requires: opencv-python >= 4.8 (FaceDetectorYN, DIS), numpy, ffmpeg (only when the frame rates differ).
"""
from __future__ import annotations
import argparse, json, math, pathlib, subprocess, sys, tempfile, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
YUNET = HERE.parents[1] / "models" / "yunet" / "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
# canonical face: eyes on a horizontal line, inter-ocular distance IOD, crop W x H (in IOD units below)
CANON = {"eye_y": 1.15, "nose_y": 1.75, "w": 2.6, "h": 3.3}
REGION_CLASSES = ("ls_core", "switch", "orig")


# ------------------------------------------------------------------------------------------------ video io
def probe_fps(p: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", p],
                         capture_output=True, text=True, check=True).stdout.strip()
    n, d = out.split("/") if "/" in out else (out, "1")
    return float(n) / float(d)


def resample_to(p: str, fps: float, tmpdir: str) -> str:
    out = str(pathlib.Path(tmpdir) / "reference_resampled.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", p, "-vf", f"fps={fps}", "-an", "-c:v", "libx264", "-crf", "12", out], check=True)
    return out


# ------------------------------------------------------------------------------------------------ landmarks / stabilisation
class YuNet:
    def __init__(self, path: pathlib.Path, score: float = 0.6):
        import cv2
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["curl", "-fsSL", "-o", str(path), YUNET_URL], check=True)
        self.det = cv2.FaceDetectorYN.create(str(path), "", (320, 320), score, 0.3, 50); self.size = None

    def __call__(self, frame) -> np.ndarray | None:
        """5 landmarks (right eye, left eye, nose, right mouth corner, left mouth corner) in image coordinates (image-left first for
        the eyes after sorting), plus bbox, of the largest face; None if no face."""
        h, w = frame.shape[:2]
        if self.size != (w, h): self.det.setInputSize((w, h)); self.size = (w, h)
        _, faces = self.det.detect(frame)
        if faces is None or len(faces) == 0: return None
        f = max(faces, key=lambda r: r[2] * r[3])
        pts = f[4:14].reshape(5, 2).astype(np.float64)
        eyes = pts[:2][np.argsort(pts[:2, 0])]; mouth = pts[3:5][np.argsort(pts[3:5, 0])]
        return np.concatenate([eyes, pts[2:3], mouth, f[None, 0:2], f[None, 0:2] + f[None, 2:4]])   # 7x2: eL eR nose mL mR bb0 bb1


def gauss_smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    """NaN-aware Gaussian smoothing along axis 0 (T, ...)."""
    if sigma <= 0: return a
    r = int(math.ceil(3 * sigma)); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    flat = a.reshape(len(a), -1); out = np.full_like(flat, np.nan)
    ok = np.isfinite(flat).all(1)
    for t in range(len(a)):
        lo, hi = max(0, t - r), min(len(a), t + r + 1); m = ok[lo:hi]
        if not ok[t] or m.sum() == 0: continue
        w = k[r - (t - lo): r + (hi - t)][m]; out[t] = (flat[lo:hi][m] * w[:, None]).sum(0) / w.sum()
    return out.reshape(a.shape)


def similarity(src3: np.ndarray, iod: float) -> np.ndarray:
    """2x3 similarity mapping (left eye, right eye, nose) to the canonical template (least squares, 4 dof)."""
    import cv2
    W = CANON["w"] * iod; cx = W / 2
    dst = np.array([[cx - iod / 2, CANON["eye_y"] * iod], [cx + iod / 2, CANON["eye_y"] * iod], [cx, CANON["nose_y"] * iod]], np.float64)
    M, _ = cv2.estimateAffinePartial2D(src3.astype(np.float32), dst.astype(np.float32), method=cv2.LMEDS)
    return M


# ------------------------------------------------------------------------------------------------ manifest helpers
def ls_layout(man: dict, n: int, win: int, ls_window: int = 16):
    segs = sorted([s for s in man["lipsync"]["segments"] if s["action"] != "SPLIT"], key=lambda s: s["start_frame"])
    ls = np.zeros(n, bool); phase = np.full(n, -1)
    for s in segs:
        if s["action"] == "LATENT_SYNC":
            a, b = s["start_frame"], min(n, s["end_frame"]); ls[a:b] = True; phase[a:b] = (np.arange(a, b) - a) % ls_window
    sw = [b["start_frame"] for a, b in zip(segs, segs[1:]) if (a["action"] == "LATENT_SYNC") != (b["action"] == "LATENT_SYNC")]
    near = np.zeros(n, bool)
    for f in sw: near[max(0, f - win):min(n, f + win)] = True
    return {"ls_core": ls & ~near, "switch": near, "orig": ~ls & ~near}, phase, sw


def window_boundaries(sig: dict, frames: np.ndarray, in_range: np.ndarray, cls: dict, phase: np.ndarray, man: dict, P: int) -> dict:
    """LatentSync denoises independent P-frame windows from each LS segment start: is there a jump at every window boundary?
    For the mouth: per-phase means (LS core frames) of d1 / resid / vel / jerk, phase-0 over inner-phase ratio for the video and for the
    reference on the same frames, and per boundary the d1 at the first frame of a window / median d1 of the previous window's inner frames."""
    ph = phase[frames]; core = in_range & cls["ls_core"][frames]; inner = [q for q in range(2, P - 2)]
    def prof(x): return [None if not (core & (ph == q) & np.isfinite(x)).any() else round(float(np.nanmean(x[core & (ph == q)])), 4) for q in range(P)]
    out = {"window_frames": P}
    for k in ("d1", "resid", "vel", "jerk"):
        for s_, lab in (("v", "video"), ("r", "reference")):
            pr = prof(sig[k]["mouth"][s_]); inn = [pr[q] for q in inner if pr[q] is not None]
            out.setdefault(k, {})[lab] = {"by_phase": pr, "phase0_over_inner": round(pr[0] / float(np.mean(inn)), 3) if pr[0] and inn else None}
    pos = {int(f): j for j, f in enumerate(frames)}; jv, jr = [], []
    for sgm in man["lipsync"]["segments"]:
        if sgm["action"] != "LATENT_SYNC": continue
        for w in range(sgm["start_frame"] + P, sgm["end_frame"], P):
            j = pos.get(w)
            if j is None or j < P or not in_range[j]: continue
            for s_, lst in (("v", jv), ("r", jr)):
                x = sig["d1"]["mouth"][s_]; base = np.nanmedian(x[j - P + 2:j - 1])
                if np.isfinite(x[j]) and base > 0: lst.append(float(x[j] / base))
    for lab, lst in (("video", jv), ("reference", jr)):
        a_ = np.array(lst) if lst else np.array([np.nan])
        out.setdefault("boundary_d1_jump", {})[lab] = {"boundaries": len(lst), "median": round(float(np.nanmedian(a_)), 3), "p75": round(float(np.nanpercentile(a_, 75)), 3) if lst else None,
                                                        "share_gt_1_5": round(float(np.mean(a_ > 1.5)), 3) if lst else None, "share_gt_2": round(float(np.mean(a_ > 2.0)), 3) if lst else None}
    return out


# ------------------------------------------------------------------------------------------------ main measurement
def iter_frames(p: str, keep: set):
    """Yield (index, BGR frame) for the indices in `keep`, streaming (no whole-video buffer)."""
    import cv2
    cap = cv2.VideoCapture(p); i = 0; last = max(keep)
    while i <= last:
        ok, f = cap.read()
        if not ok: break
        if i in keep: yield i, f
        i += 1
    cap.release()


def measure(video: str, reference: str, ranges: list, iod: float, lm_sigma: float, sheet_dir: str | None, tag: str, hp_sigma: float = 4.0) -> dict:
    import cv2
    need = set()
    for a, b in ranges: need.update(range(max(0, a - 2), b + 2))
    yn = YuNet(YUNET); T0, T1 = min(need), max(need); T = T1 - T0 + 1
    L = np.full((T, 7, 2), np.nan)
    for i, f in iter_frames(reference, need):                                  # pass 1: reference landmarks only
        p = yn(f)
        if p is not None: L[i - T0] = p
    det_rate = float(np.isfinite(L[[i - T0 for i in need], 0, 0]).mean())
    Ls = gauss_smooth(L, lm_sigma)
    Ms = {}; mouth_c = []
    for i in sorted(need):
        t = i - T0
        if not np.isfinite(Ls[t, :3]).all(): continue
        M = similarity(Ls[t, :3], iod); Ms[i] = M
        if np.isfinite(Ls[t, 3:5]).all(): mouth_c.append(np.c_[Ls[t, 3:5], np.ones(2)] @ M.T)
    if not Ms or not mouth_c: raise SystemExit("no face found in the reference frames")
    mc = np.median(np.stack(mouth_c), axis=0); mx, my = mc.mean(0); mw = float(np.linalg.norm(mc[1] - mc[0]))
    W, H = int(round(CANON["w"] * iod)), int(round(CANON["h"] * iod))
    def box(x0, y0, x1, y1): return [int(max(0, round(x0))), int(max(0, round(y0))), int(min(W, round(x1))), int(min(H, round(y1)))]
    rois = {"mouth": box(mx - 0.85 * mw, my - 0.55 * mw, mx + 0.85 * mw, my + 0.6 * mw),
            "upper": box(W / 2 - 1.15 * iod, (CANON["eye_y"] - 0.6) * iod, W / 2 + 1.15 * iod, (CANON["eye_y"] + 0.35) * iod)}
    lf = box(mx - 1.45 * mw, my - 0.85 * mw, mx + 1.45 * mw, min(H, my + 1.35 * mw))
    masks = {}
    for name in ("mouth", "upper"):
        m = np.zeros((H, W), bool); b = rois[name]; m[b[1]:b[3], b[0]:b[2]] = True; masks[name] = m
    lower = np.zeros((H, W), bool); lower[lf[1]:lf[3], lf[0]:lf[2]] = True; lower &= ~masks["mouth"]; masks["lowerface"] = lower   # ring around the mouth box
    crops = {"v": {}, "r": {}}
    for k, src in (("v", video), ("r", reference)):                            # pass 2: canonical grey crops of both videos, same transform
        for i, f in iter_frames(src, set(Ms)):
            crops[k][i] = cv2.cvtColor(cv2.warpAffine(f, Ms[i], (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    # rigid head region (brows, eyes, nose bridge) for the per-frame head-motion model: the landmark alignment is temporally smoothed, so
    # hand-held camera shake / fast head motion survives in the crops; it is common to both videos and would swamp the mouth signal
    rig = np.zeros((H, W), bool); b = rois["upper"]; rig[b[1]:b[3], b[0]:b[2]] = True
    rig[int(CANON["eye_y"] * iod):int((CANON["nose_y"] + 0.1) * iod), int(W / 2 - 0.45 * iod):int(W / 2 + 0.45 * iod)] = True
    ry, rx = np.nonzero(rig[::4, ::4]); ry, rx = ry * 4, rx * 4; rpts = np.stack([rx, ry], 1).astype(np.float32)
    def head_field(F):
        """Similarity motion of the rigid head region fitted to the flow (RANSAC), evaluated on the whole crop."""
        Mh, _ = cv2.estimateAffinePartial2D(rpts, rpts + F[ry, rx], method=cv2.RANSAC, ransacReprojThreshold=1.0)
        if Mh is None: return np.zeros_like(F)
        return np.stack([Mh[0, 0] * gx + Mh[0, 1] * gy + Mh[0, 2] - gx, Mh[1, 0] * gx + Mh[1, 1] * gy + Mh[1, 2] - gy], -1)
    # stage-2 stabilisation (complementary filter): the landmark path is smooth (low frequencies); the high-frequency head / camera shake
    # left in the crops is measured on the REFERENCE crops (similarity fit of the rigid-region flow, chained per contiguous run), high-passed
    # (Gaussian sigma --hp-sigma) and removed from BOTH videos with the same correction warp.
    corr = {}
    if hp_sigma > 0:
        idx = sorted(crops["r"]); runs = []; cur = [idx[0]]
        for i in idx[1:]:
            if i == cur[-1] + 1: cur.append(i)
            else: runs.append(cur); cur = [i]
        runs.append(cur)
        cx, cy = W / 2, CANON["eye_y"] * iod
        for run in runs:
            P = np.zeros((len(run), 4))                                          # cumulative (tx, ty, rot, log-scale) about (cx, cy)
            for j in range(1, len(run)):
                F = dis.calc(crops["r"][run[j - 1]], crops["r"][run[j]], None)
                Mh, _ = cv2.estimateAffinePartial2D(rpts, rpts + F[ry, rx], method=cv2.RANSAC, ransacReprojThreshold=1.0)
                if Mh is None: P[j] = P[j - 1]; continue
                a_, b_ = Mh[0, 0], Mh[1, 0]; th = math.atan2(b_, a_); sc = math.log(max(math.hypot(a_, b_), 1e-6))
                dxc = Mh[0, 0] * cx + Mh[0, 1] * cy + Mh[0, 2] - cx; dyc = Mh[1, 0] * cx + Mh[1, 1] * cy + Mh[1, 2] - cy
                P[j] = P[j - 1] + [dxc, dyc, th, sc]
            Q = P - gauss_smooth(P, hp_sigma) if len(run) > 3 else np.zeros_like(P)
            for j, i in enumerate(run):
                dxc, dyc, th, sc = Q[j]; c, s_ = math.cos(-th) / math.exp(sc), math.sin(-th) / math.exp(sc)
                # inverse of the residual similarity about (cx, cy): x' = R^-1 (x - (cx, cy) - d) + (cx, cy)
                A = np.array([[c, -s_], [s_, c]]); t = np.array([cx, cy]) - A @ (np.array([cx, cy]) + [dxc, dyc])
                corr[i] = np.c_[A, t]
        for k in ("v", "r"):
            for i, M in corr.items():
                if i in crops[k]: crops[k][i] = cv2.warpAffine(crops[k][i], M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    keys = ("vel", "jerk", "resid", "acc", "d1", "sharp", "shear", "shear95")
    sig = {k: {r: {s: np.full(T, np.nan) for s in ("v", "r")} for r in masks} for k in keys}
    for s in ("v", "r"):
        C = crops[s]; Fw = {}
        for i in sorted(C):
            if i - 1 in C:
                a, b = C[i - 1], C[i]; F = dis.calc(a, b, None); Fb = dis.calc(b, a, None)
                Fw[i] = Frel = F - head_field(F)                                # motion relative to the head
                # on the grid of I(t): Fb pulls I(t-1) onto it; forward-backward check there (F sampled at x + Fb(x) must be ~ -Fb(x))
                fx = cv2.remap(F[..., 0], gx + Fb[..., 0], gy + Fb[..., 1], cv2.INTER_LINEAR); fy = cv2.remap(F[..., 1], gx + Fb[..., 0], gy + Fb[..., 1], cv2.INTER_LINEAR)
                ok = np.hypot(Fb[..., 0] + fx, Fb[..., 1] + fy) < 0.5 + 0.05 * np.hypot(Fb[..., 0], Fb[..., 1])
                warped = cv2.remap(a.astype(np.float32), gx + Fb[..., 0], gy + Fb[..., 1], cv2.INTER_LINEAR)
                res = np.abs(b.astype(np.float32) - warped); mag = np.hypot(Frel[..., 0], Frel[..., 1])
                # flow shear: spatial gradient of the head-relative flow (a seam where pasted and source skin move differently)
                Fs = cv2.GaussianBlur(Frel, (0, 0), 1.5)
                shr = sum(np.abs(cv2.Sobel(Fs[..., c], cv2.CV_32F, dx, dy, ksize=3)) for c in (0, 1) for dx, dy in ((1, 0), (0, 1))) / 8.0
                for r, m in masks.items():
                    mm = m & ok
                    sig["vel"][r][s][i - T0] = float(mag[m].mean())
                    sig["shear"][r][s][i - T0] = float(shr[m].mean()); sig["shear95"][r][s][i - T0] = float(np.percentile(shr[m], 95))
                    sig["resid"][r][s][i - T0] = float(res[mm].mean()) if mm.sum() > 20 else np.nan
                if i - 1 in Fw:                                                 # jerk centred on frame i-1: F(i-1->i) - F(i-2->i-1)
                    d = np.hypot(Frel[..., 0] - Fw[i - 1][..., 0], Frel[..., 1] - Fw[i - 1][..., 1])
                    for r, m in masks.items(): sig["jerk"][r][s][i - 1 - T0] = float(d[m].mean())
                Fw.pop(i - 2, None)
            lap = np.abs(cv2.Laplacian(C[i].astype(np.float32), cv2.CV_32F, ksize=3))
            for r, m in masks.items(): sig["sharp"][r][s][i - T0] = float(lap[m].mean())
            if i - 1 in C:
                d1 = np.abs(C[i].astype(np.float32) - C[i - 1].astype(np.float32))
                for r, m in masks.items(): sig["d1"][r][s][i - T0] = float(d1[m].mean())
            if i - 1 in C and i - 2 in C:
                acc = np.abs(C[i].astype(np.float32) - 2 * C[i - 1].astype(np.float32) + C[i - 2].astype(np.float32))
                for r, m in masks.items(): sig["acc"][r][s][i - 1 - T0] = float(acc[m].mean())
    if sheet_dir:
        write_sheets(sheet_dir, tag, crops, rois, lf, ranges)
    return {"sig": sig, "frames": np.arange(T0, T1 + 1), "rois": rois, "lowerface_box": lf, "canvas": [W, H], "iod": iod,
            "det_rate_reference": round(det_rate, 4), "mouth_width_canon_px": round(mw, 1)}


def write_sheets(d: str, tag: str, crops: dict, rois: dict, lf, ranges: list, per_range: int = 16):
    """Contact sheet per range: rows = video / reference lower face crops for up to `per_range` consecutive frames from the range start."""
    import cv2
    out = pathlib.Path(d); out.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = lf
    for a, b in ranges:
        rows = []
        for s in ("v", "r"):
            tiles = []
            for i in range(a, min(b, a + per_range)):
                if i not in crops[s]: continue
                t = cv2.resize(crops[s][i][y0:y1, x0:x1], None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                cv2.putText(t, str(i), (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1, cv2.LINE_AA); tiles.append(t)
            if tiles: rows.append(np.hstack(tiles))
        if len(rows) == 2:
            cv2.imwrite(str(out / f"{tag}_lowerface_{a:05d}-{b:05d}.png"), np.vstack(rows))


def aggregate(sig: dict, sel: np.ndarray) -> dict:
    out = {}
    for k, regs in sig.items():
        out[k] = {}
        for r, sv in regs.items():
            v, ref = sv["v"], sv["r"]; m = sel & np.isfinite(v) & np.isfinite(ref)
            if m.sum() < 5: out[k][r] = None; continue
            mv, mr = float(np.mean(v[m])), float(np.mean(ref[m]))
            out[k][r] = {"video": round(mv, 4), "reference": round(mr, 4), "ratio": round(mv / max(mr, 1e-9), 3),
                         "p95_ratio": round(float(np.percentile(v[m], 95) / max(np.percentile(ref[m], 95), 1e-9)), 3), "frames": int(m.sum())}
    tw = {}
    for r in ("mouth", "lowerface", "upper"):
        j, vl = sig["jerk"][r], sig["vel"][r]
        m = sel & np.isfinite(j["v"]) & np.isfinite(j["r"]) & np.isfinite(vl["v"]) & np.isfinite(vl["r"])
        if m.sum() < 5: tw[r] = None; continue
        tv, tr = float(j["v"][m].sum() / max(vl["v"][m].sum(), 1e-9)), float(j["r"][m].sum() / max(vl["r"][m].sum(), 1e-9))
        tw[r] = {"video": round(tv, 4), "reference": round(tr, 4), "ratio": round(tv / max(tr, 1e-9), 3)}
    out["twitch_index"] = tw
    imp, pops = {}, {}
    for r in ("mouth", "lowerface", "upper"):
        A_, D_ = sig["acc"][r], sig["d1"][r]
        m = sel & np.isfinite(A_["v"]) & np.isfinite(A_["r"]) & np.isfinite(D_["v"]) & np.isfinite(D_["r"])
        if m.sum() < 20: imp[r] = pops[r] = None; continue
        iv, ir = float(A_["v"][m].sum() / max(D_["v"][m].sum(), 1e-9)), float(A_["r"][m].sum() / max(D_["r"][m].sum(), 1e-9))
        imp[r] = {"video": round(iv, 4), "reference": round(ir, 4), "ratio": round(iv / max(ir, 1e-9), 3)}
        thr_d, thr_a = np.percentile(D_["r"][m], 98), np.percentile(A_["r"][m], 98)
        pops[r] = {"d1_above_ref_p98": round(float((D_["v"][m] > thr_d).mean()) / 0.02, 2), "acc_above_ref_p98": round(float((A_["v"][m] > thr_a).mean()) / 0.02, 2)}
    out["impulse_index"] = imp; out["pop_rate_vs_reference"] = pops
    return out


def parse_ranges(items, n):
    if not items: return [(1, n - 2)]
    out = []
    for it in items:
        a, b = it.split(":"); out.append((max(1, int(a)), min(n - 2, int(b))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="candidate video (the dubbed output)")
    ap.add_argument("--reference", required=True, help="reference video with the same frames (the 25 fps master or the source; resampled to the video fps if needed)")
    ap.add_argument("--frames", nargs="*", default=None, help="frame ranges a:b (end exclusive) in video frame indices; default = all")
    ap.add_argument("--manifest", default=None, help="run manifest (lipsync.segments) -> ls_core / switch / orig classes and LatentSync window phase")
    ap.add_argument("--out", required=True, help="JSON output"); ap.add_argument("--tag", default="video")
    ap.add_argument("--iod", type=float, default=96.0, help="canonical inter-ocular distance in px (crop = 2.6 x 3.3 IOD)")
    ap.add_argument("--lm-sigma", type=float, default=1.5, help="temporal Gaussian sigma (frames) of the reference landmarks (stabilisation only)")
    ap.add_argument("--hp-sigma", type=float, default=4.0, help="stage-2 stabilisation: high-pass sigma (frames) of the flow-measured head shake removed from both videos (0 = off)")
    ap.add_argument("--ls-window", type=int, default=16, help="LatentSync temporal window length in frames (stage2_512: num_frames = 16)")
    ap.add_argument("--win", type=int, default=5, help="+- frames around an LS<->original switch = 'switch' class")
    ap.add_argument("--series", action="store_true", help="also write the per-frame series"); ap.add_argument("--sheet", default=None, help="dir for lower-face contact sheets per range")
    a = ap.parse_args()
    t0 = time.perf_counter()
    import cv2
    fv, fr = probe_fps(a.video), probe_fps(a.reference); ref = a.reference; tmp = tempfile.mkdtemp()
    if abs(fv - fr) > 0.01: ref = resample_to(a.reference, fv, tmp)
    n = int(cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FRAME_COUNT)); nr = int(cv2.VideoCapture(ref).get(cv2.CAP_PROP_FRAME_COUNT)); n = min(n, nr)
    ranges = parse_ranges(a.frames, n)
    res = measure(a.video, ref, ranges, a.iod, a.lm_sigma, a.sheet, a.tag, a.hp_sigma)
    frames = res["frames"]; T = len(frames)
    in_range = np.zeros(T, bool)
    for x, y in ranges: in_range[(frames >= x) & (frames < y)] = True
    report = {"video": a.video, "reference": a.reference, "fps": fv, "ranges": ranges, "tag": a.tag,
              "params": {"iod": a.iod, "lm_sigma": a.lm_sigma, "win": a.win, "landmarks": "YuNet 2023mar (reference frames)", "flow": "OpenCV DIS medium"},
              "rois_canonical": res["rois"], "lowerface_box": res["lowerface_box"], "canvas": res["canvas"], "det_rate_reference": res["det_rate_reference"],
              "all": aggregate(res["sig"], in_range), "per_range": {}}
    for x, y in ranges:
        report["per_range"][f"{x}:{y}"] = aggregate(res["sig"], (frames >= x) & (frames < y))
    if a.manifest:
        man = json.loads(pathlib.Path(a.manifest).read_text())
        cls, phase, sw = ls_layout(man, int(frames[-1]) + 2, a.win, a.ls_window)
        report["classes"] = {c: aggregate(res["sig"], in_range & cls[c][frames]) for c in REGION_CLASSES}
        report["latentsync_window_boundaries"] = window_boundaries(res["sig"], frames, in_range, cls, phase, man, a.ls_window)
        report["switch_frames"] = sw
    # worst 1 s windows (mouth jerk ratio), for review clips
    jv, jr = res["sig"]["jerk"]["mouth"]["v"], res["sig"]["jerk"]["mouth"]["r"]; win = []
    med_r = np.nanmedian(jr[in_range]) if np.isfinite(jr[in_range]).any() else 1.0
    for s in range(0, T - 25, 12):
        sl = slice(s, s + 25); m = in_range[sl] & np.isfinite(jv[sl])
        if m.sum() >= 15: win.append((float(np.nanmean(jv[sl][m]) / max(med_r, 1e-6)), int(frames[s])))
    win.sort(reverse=True); worst = []
    for v, f in win:
        if all(abs(f - w["frame"]) > 25 for w in worst): worst.append({"frame": f, "t": round(f / fv, 2), "mouth_jerk_vs_reference_median": round(v, 2)})
        if len(worst) >= 8: break
    report["worst_windows"] = worst
    # per second: the numbers a reviewer's "twitch at t s" can be checked against, plus a rank per column (1 = worst)
    sg = res["sig"]; secs = []
    for t_ in range(int(frames[0] // fv), int(frames[-1] // fv) + 1):
        m = in_range & (frames >= t_ * fv) & (frames < (t_ + 1) * fv)
        if m.sum() < 10: continue
        def mr(k, reg, f=np.nanmean):
            v, r_ = sg[k][reg]["v"][m], sg[k][reg]["r"][m]
            return float(f(v) / max(f(r_), 1e-6)) if np.isfinite(v).any() and np.isfinite(r_).any() else None
        vv, vr = np.nanmean(sg["vel"]["mouth"]["v"][m]), np.nanmean(sg["vel"]["mouth"]["r"][m])
        lv = np.nanmean(sg["vel"]["lowerface"]["v"][m])
        secs.append({"t": t_, "resid_mouth": mr("resid", "mouth"), "d1_mouth": mr("d1", "mouth"), "impulse_mouth": None,
                     "shear95_lowerface": mr("shear95", "lowerface"), "shear_lowerface": mr("shear", "lowerface"),
                     # frozen mouth on a talking face: the surrounding face moves (lower-face ring of the OUTPUT) while the output mouth does not,
                     # relative to the same proportion in the reference: > 1 = the mouth is stiller than the face around it, compared with the source
                     "frozen_mouth": float((lv / max(vv, 1e-3)) / max(np.nanmean(sg["vel"]["lowerface"]["r"][m]) / max(vr, 1e-3), 1e-6)) if np.isfinite(vv) and np.isfinite(lv) else None,
                     "max_d1_jump": float(np.nanmax(sg["d1"]["mouth"]["v"][m]) / max(np.nanmedian(sg["d1"]["mouth"]["v"][in_range]), 1e-6)) if np.isfinite(sg["d1"]["mouth"]["v"][m]).any() else None})
    for k in ("resid_mouth", "shear95_lowerface", "frozen_mouth", "max_d1_jump", "d1_mouth"):
        order = sorted([x for x in secs if x[k] is not None], key=lambda x: -x[k])
        for rank, x in enumerate(order, 1): x[f"rank_{k}"] = rank
    report["per_second"] = secs
    if a.series:
        report["series"] = {"frames": frames.tolist(), **{f"{k}_{r}_{s}": [None if not np.isfinite(x) else round(float(x), 4) for x in res["sig"][k][r][s]]
                                                         for k in res["sig"] for r in res["sig"][k] for s in ("v", "r")}}
    report["seconds"] = round(time.perf_counter() - t0, 1)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.out).write_text(json.dumps(report, indent=1))
    A = report["all"]
    def r(k, reg): x = A[k][reg]; return "-" if x is None else f"{x['ratio']:.2f}"
    print(f"== {a.tag}: frames {int(in_range.sum())}  det {res['det_rate_reference']:.3f}  ({report['seconds']} s)")
    print(f"   ratio video/reference  mouth: jerk {r('jerk','mouth')} vel {r('vel','mouth')} resid {r('resid','mouth')} acc {r('acc','mouth')} twitch {A['twitch_index']['mouth']['ratio'] if A['twitch_index']['mouth'] else '-'}"
          f" | lowerface: jerk {r('jerk','lowerface')} resid {r('resid','lowerface')} | upper: jerk {r('jerk','upper')} resid {r('resid','upper')} acc {r('acc','upper')}")
    im, po = A["impulse_index"], A["pop_rate_vs_reference"]
    print(f"   impulse ratio mouth {im['mouth']['ratio'] if im['mouth'] else '-'} lowerface {im['lowerface']['ratio'] if im['lowerface'] else '-'} upper {im['upper']['ratio'] if im['upper'] else '-'}"
          f" | pops (ref = 1) mouth d1 {po['mouth']['d1_above_ref_p98'] if po['mouth'] else '-'} acc {po['mouth']['acc_above_ref_p98'] if po['mouth'] else '-'}")
    if "latentsync_window_boundaries" in report:
        p = report["latentsync_window_boundaries"]; bj = p["boundary_d1_jump"]
        print(f"   LS window boundary (phase 0 / inner), mouth d1: video {p['d1']['video']['phase0_over_inner']} ref {p['d1']['reference']['phase0_over_inner']}; "
              f"resid: video {p['resid']['video']['phase0_over_inner']} ref {p['resid']['reference']['phase0_over_inner']}; "
              f"boundaries with a d1 jump > 1.5x: video {bj['video']['share_gt_1_5']} ref {bj['reference']['share_gt_1_5']} (n={bj['video']['boundaries']})")
    if "classes" in report and report["classes"]["ls_core"]["resid"]["mouth"]:
        c = report["classes"]["ls_core"]; print(f"   LS core mouth: resid {c['resid']['mouth']['ratio']} vel {c['vel']['mouth']['ratio']} d1 {c['d1']['mouth']['ratio']} impulse {c['impulse_index']['mouth']['ratio'] if c['impulse_index']['mouth'] else '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
